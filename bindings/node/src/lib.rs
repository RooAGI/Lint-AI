use std::{path::Path, sync::Mutex, time::Duration};

use membrick::{
    AddRequest, GetRequest, ListRequest, MemoryService, PipelineOptions, SearchRequest,
    Tier1NerProvider, UpdateRequest,
};
use napi::bindgen_prelude::*;
use napi_derive::napi;
use reqwest::{blocking::Client, Method, StatusCode};
use serde::{de::DeserializeOwned, Serialize};
use serde_json::Value;

#[napi(object)]
pub struct MemoryOptions {
    pub path: Option<String>,
    pub base_url: Option<String>,
    pub api_key: Option<String>,
    pub ner_provider: Option<String>,
    pub spacy_model: Option<String>,
}

enum Backend {
    Local(MemoryService),
    Remote(RemoteMemoryClient),
}

#[napi]
pub struct Memory {
    backend: Mutex<Backend>,
}

#[napi]
impl Memory {
    #[napi(constructor)]
    pub fn new(options: Option<MemoryOptions>) -> Result<Self> {
        let options = options.unwrap_or(MemoryOptions {
            path: None,
            base_url: None,
            api_key: None,
            ner_provider: None,
            spacy_model: None,
        });
        if options.path.is_some() && options.base_url.is_some() {
            return Err(Error::from_reason("path and base_url cannot both be set"));
        }

        let backend = if let Some(base_url) = options.base_url {
            Backend::Remote(RemoteMemoryClient::new(base_url, options.api_key)?)
        } else {
            let mut pipeline = PipelineOptions::default();
            match options.ner_provider.as_deref().unwrap_or("heuristic") {
                "heuristic" => pipeline.ner_provider = Tier1NerProvider::Heuristic,
                "spacy" => pipeline.ner_provider = Tier1NerProvider::Spacy,
                other => {
                    return Err(Error::from_reason(format!(
                        "ner_provider must be 'heuristic' or 'spacy' (got '{other}')"
                    )))
                }
            }
            if let Some(model) = options.spacy_model {
                pipeline.spacy_model = model;
            }
            let service = match options.path {
                Some(path) => {
                    MemoryService::at_path(Path::new(&path), pipeline).map_err(runtime_error)?
                }
                None => MemoryService::in_memory(pipeline),
            };
            Backend::Local(service)
        };
        Ok(Self {
            backend: Mutex::new(backend),
        })
    }

    /// Add one memory. Arguments mirror the Python Memory.add method.
    #[napi]
    pub async fn add(
        &self,
        request_id: String,
        user_id: String,
        session_id: String,
        messages: Vec<Value>,
    ) -> Result<Value> {
        let messages = messages
            .into_iter()
            .map(from_value)
            .collect::<Result<Vec<_>>>()?;
        let request = AddRequest {
            request_id,
            user_id,
            session_id,
            messages,
        };
        let mut backend = self.backend.lock().map_err(lock_error)?;
        match &mut *backend {
            Backend::Local(service) => to_value(service.add(request).map_err(runtime_error)?),
            Backend::Remote(client) => client.add(&request),
        }
    }

    /// Add between 1 and 128 memories. Remote mode sends sequential requests,
    /// matching Python's fallback because the HTTP API has no batch route.
    #[napi]
    pub async fn add_batch(&self, requests: Vec<Value>) -> Result<Vec<Value>> {
        if requests.is_empty() || requests.len() > 128 {
            return Err(Error::from_reason(
                "requests must contain between 1 and 128 entries",
            ));
        }
        let requests = requests
            .into_iter()
            .map(from_value::<AddRequest>)
            .collect::<Result<Vec<_>>>()?;
        let mut backend = self.backend.lock().map_err(lock_error)?;
        match &mut *backend {
            Backend::Local(service) => {
                to_values(service.add_batch(requests).map_err(runtime_error)?)
            }
            Backend::Remote(client) => requests.iter().map(|r| client.add(r)).collect(),
        }
    }

    /// Search memories for a user. Optional request fields are session_id,
    /// scope, filters, and reference_date.
    #[napi]
    pub async fn search(
        &self,
        query: String,
        user_id: String,
        top_k: Option<u32>,
        session_id: Option<String>,
        filters: Option<Value>,
        scope: Option<String>,
    ) -> Result<Vec<Value>> {
        let filters = match filters {
            Some(Value::Null) | None => None,
            Some(value) => Some(from_value::<std::collections::BTreeMap<String, String>>(
                value,
            )?),
        };
        let request = SearchRequest {
            query,
            options: None,
            user_id,
            top_k: top_k.unwrap_or(10) as usize,
            session_id,
            scope,
            filters,
            reference_date: None,
        };
        let mut backend = self.backend.lock().map_err(lock_error)?;
        match &mut *backend {
            Backend::Local(service) => {
                let response = service.search(request).map_err(runtime_error)?;
                to_values(response.data)
            }
            Backend::Remote(client) => {
                let response = client
                    .request(Method::POST, "/v1/memories/search")
                    .json(&request)
                    .send()
                    .map_err(runtime_error)?;
                let parsed: membrick::SearchResponse = decode(response)?;
                to_values(parsed.data)
            }
        }
    }

    #[napi]
    pub async fn get(
        &self,
        memory_id: String,
        user_id: String,
        include_inactive: Option<bool>,
    ) -> Result<Option<Value>> {
        let request = GetRequest {
            user_id,
            memory_id,
            include_inactive: include_inactive.unwrap_or(false),
        };
        let backend = self.backend.lock().map_err(lock_error)?;
        let result = match &*backend {
            Backend::Local(service) => service.get(request).map_err(runtime_error)?,
            Backend::Remote(client) => client.get(&request)?,
        };
        result.map(to_value).transpose()
    }

    #[napi]
    pub async fn list(
        &self,
        user_id: String,
        session_id: Option<String>,
        limit: Option<u32>,
        cursor: Option<String>,
        include_inactive: Option<bool>,
    ) -> Result<Value> {
        let request = ListRequest {
            user_id,
            session_id,
            limit: limit.unwrap_or(100) as usize,
            cursor,
            include_inactive: include_inactive.unwrap_or(false),
        };
        let backend = self.backend.lock().map_err(lock_error)?;
        let result = match &*backend {
            Backend::Local(service) => service.list(request).map_err(runtime_error)?,
            Backend::Remote(client) => client.list(&request)?,
        };
        to_value(result)
    }

    #[napi]
    pub async fn update(
        &self,
        memory_id: String,
        user_id: String,
        content: String,
        role: Option<String>,
        timestamp: Option<f64>,
        expires_at_ms: Option<f64>,
    ) -> Result<Option<Value>> {
        let timestamp = timestamp
            .map(|value| safe_integer(value, "timestamp"))
            .transpose()?;
        let expires_at_ms = expires_at_ms
            .map(|value| safe_integer(value, "expires_at_ms"))
            .transpose()?
            .map(|value| {
                u64::try_from(value)
                    .map_err(|_| Error::from_reason("expires_at_ms must be non-negative"))
            })
            .transpose()?;
        let request = UpdateRequest {
            memory_id,
            user_id,
            content,
            role,
            timestamp,
            expires_at_ms,
        };
        let mut backend = self.backend.lock().map_err(lock_error)?;
        let result = match &mut *backend {
            Backend::Local(service) => service.update(request).map_err(runtime_error)?,
            Backend::Remote(client) => client.update(&request)?,
        };
        result.map(to_value).transpose()
    }

    #[napi]
    pub async fn delete(&self, user_id: String, memory_id: String) -> Result<bool> {
        let mut backend = self.backend.lock().map_err(lock_error)?;
        match &mut *backend {
            Backend::Local(service) => service.delete(&user_id, &memory_id).map_err(runtime_error),
            Backend::Remote(client) => client.delete(&user_id, &memory_id),
        }
    }

    #[napi]
    pub async fn refresh(&self) -> Result<()> {
        let mut backend = self.backend.lock().map_err(lock_error)?;
        match &mut *backend {
            Backend::Local(service) => service.refresh().map_err(runtime_error),
            Backend::Remote(client) => client.refresh(),
        }
    }
}

struct RemoteMemoryClient {
    client: Client,
    base_url: String,
    api_key: Option<String>,
}

impl RemoteMemoryClient {
    fn new(base_url: String, api_key: Option<String>) -> Result<Self> {
        let base_url = base_url.trim_end_matches('/').to_owned();
        if base_url.is_empty() {
            return Err(Error::from_reason("base_url must not be empty"));
        }
        let client = Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .map_err(runtime_error)?;
        Ok(Self {
            client,
            base_url,
            api_key,
        })
    }

    fn request(&self, method: Method, path: &str) -> reqwest::blocking::RequestBuilder {
        let request = self
            .client
            .request(method, format!("{}{}", self.base_url, path));
        match &self.api_key {
            Some(key) => request.bearer_auth(key),
            None => request,
        }
    }

    fn add(&self, request: &AddRequest) -> Result<Value> {
        let response = self
            .request(Method::POST, "/v1/memories")
            .json(request)
            .send()
            .map_err(runtime_error)?;
        decode(response)
    }

    fn get(&self, request: &GetRequest) -> Result<Option<membrick::MemoryRecord>> {
        let response = self
            .request(
                Method::GET,
                &format!("/v1/memories/{}", encode_path_segment(&request.memory_id)),
            )
            .query(&[
                ("user_id", request.user_id.as_str()),
                (
                    "include_inactive",
                    if request.include_inactive {
                        "true"
                    } else {
                        "false"
                    },
                ),
            ])
            .send()
            .map_err(runtime_error)?;
        if response.status() == StatusCode::NOT_FOUND {
            return Ok(None);
        }
        decode(response)
    }

    fn list(&self, request: &ListRequest) -> Result<membrick::ListResponse> {
        let response = self
            .request(Method::GET, "/v1/memories")
            .query(request)
            .send()
            .map_err(runtime_error)?;
        decode(response)
    }

    fn update(&self, request: &UpdateRequest) -> Result<Option<membrick::MemoryRecord>> {
        let response = self
            .request(
                Method::PATCH,
                &format!("/v1/memories/{}", encode_path_segment(&request.memory_id)),
            )
            .json(request)
            .send()
            .map_err(runtime_error)?;
        if response.status() == StatusCode::NOT_FOUND {
            return Ok(None);
        }
        decode(response)
    }

    fn delete(&self, user_id: &str, memory_id: &str) -> Result<bool> {
        let response = self
            .request(
                Method::DELETE,
                &format!("/v1/memories/{}", encode_path_segment(memory_id)),
            )
            .query(&[("user_id", user_id)])
            .send()
            .map_err(runtime_error)?;
        if response.status() == StatusCode::NOT_FOUND {
            return Ok(false);
        }
        check_status(response.status(), "remote memory deletion")?;
        Ok(true)
    }

    fn refresh(&self) -> Result<()> {
        let response = self
            .request(Method::POST, "/v1/memories/refresh")
            .send()
            .map_err(runtime_error)?;
        check_status(response.status(), "remote memory refresh")
    }
}

fn decode<T: DeserializeOwned>(response: reqwest::blocking::Response) -> Result<T> {
    let status = response.status();
    let body = response.text().map_err(runtime_error)?;
    if !status.is_success() {
        return Err(Error::from_reason(format!(
            "remote memory request failed ({status}): {body}"
        )));
    }
    serde_json::from_str(&body).map_err(runtime_error)
}

fn check_status(status: StatusCode, operation: &str) -> Result<()> {
    if status.is_success() {
        Ok(())
    } else {
        Err(Error::from_reason(format!("{operation} failed ({status})")))
    }
}

fn from_value<T: DeserializeOwned>(value: Value) -> Result<T> {
    serde_json::from_value(value).map_err(runtime_error)
}

fn to_value<T: Serialize>(value: T) -> Result<Value> {
    serde_json::to_value(value).map_err(runtime_error)
}

fn to_values<T: Serialize>(values: Vec<T>) -> Result<Vec<Value>> {
    values.into_iter().map(to_value).collect()
}

fn safe_integer(value: f64, field: &str) -> Result<i64> {
    const MAX_SAFE_INTEGER: f64 = 9_007_199_254_740_991.0;
    if !value.is_finite() || value.fract() != 0.0 || value.abs() > MAX_SAFE_INTEGER {
        return Err(Error::from_reason(format!(
            "{field} must be a safe integer"
        )));
    }
    Ok(value as i64)
}

fn runtime_error(error: impl std::fmt::Display) -> Error {
    Error::from_reason(error.to_string())
}

fn lock_error<T>(_: std::sync::PoisonError<T>) -> Error {
    Error::from_reason("memory service lock is poisoned")
}

fn encode_path_segment(value: &str) -> String {
    value
        .bytes()
        .map(|byte| match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'.' | b'_' | b'~' => {
                (byte as char).to_string()
            }
            _ => format!("%{byte:02X}"),
        })
        .collect()
}

#[napi]
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}
