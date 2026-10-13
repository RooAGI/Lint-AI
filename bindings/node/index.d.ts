export interface MemoryOptions {
  /** Persistent local store path. Omit for an in-memory local service. */
  path?: string
  /** Connect to the resident HTTP server instead of opening a local store. */
  baseUrl?: string
  /** Optional bearer token for the HTTP server. */
  apiKey?: string
  /** Local-only entity provider: heuristic (default) or spacy. */
  nerProvider?: 'heuristic' | 'spacy'
  /** Optional local-only spaCy model override. */
  spacyModel?: string
}

export interface Message {
  role: string
  content: string
  timestamp?: number | null
  expires_at_ms?: number | null
  supersedes_id?: string | null
}

export interface AddRequest {
  request_id: string
  user_id: string
  session_id: string
  messages: Message[]
}

export interface SearchRequest {
  query: string
  user_id: string
  top_k?: number
  session_id?: string | null
  scope?: string | null
  filters?: Record<string, string> | null
  reference_date?: string | null
  options?: string[] | null
}

export interface GetRequest {
  user_id: string
  memory_id: string
  include_inactive?: boolean
}

export interface ListRequest {
  user_id: string
  session_id?: string | null
  limit?: number
  cursor?: string | null
  include_inactive?: boolean
}

export interface UpdateRequest {
  user_id: string
  memory_id: string
  content: string
  role?: string | null
  timestamp?: number | null
  expires_at_ms?: number | null
}

export interface MemoryRecord {
  id: string
  content: string
  role: string | null
  user_id: string
  session_id: string | null
  created_at: string | null
  expires_at_ms: number | null
  supersedes_id: string | null
  metadata: Record<string, string>
}

export interface SearchMemory extends Partial<MemoryRecord> {
  id: string
  content: string
  score: number
  relation_evidence?: string[]
}

export interface ListResponse {
  data: MemoryRecord[]
  next_cursor: string | null
}

export interface AddResponse {
  success: boolean
  request_id: string
  user_id: string
  session_id: string
  adjudication?: unknown
}

export declare class Memory {
  constructor(options?: MemoryOptions)
  add(requestId: string, userId: string, sessionId: string, messages: Message[]): Promise<AddResponse>
  addBatch(requests: AddRequest[]): Promise<AddResponse[]>
  search(query: string, userId: string, topK?: number, sessionId?: string, filters?: Record<string, string>, scope?: string): Promise<SearchMemory[]>
  get(memoryId: string, userId: string, includeInactive?: boolean): Promise<MemoryRecord | null>
  list(userId: string, sessionId?: string, limit?: number, cursor?: string, includeInactive?: boolean): Promise<ListResponse>
  update(memoryId: string, userId: string, content: string, role?: string, timestamp?: number, expiresAtMs?: number): Promise<MemoryRecord | null>
  delete(userId: string, memoryId: string): Promise<boolean>
  refresh(): Promise<void>
}

export declare function version(): string
