use crate::graph::Graph;
use crate::report::Report;
use petgraph::visit::Dfs;
use std::collections::HashSet;

/// Report unreachable or orphan pages based on graph reachability.
pub fn check_orphans(graph: &Graph, report: &mut Report) {
    let avg_out = if graph.graph.node_count() > 0 {
        graph.graph.edge_count() as f64 / graph.graph.node_count() as f64
    } else {
        0.0
    };

    for page in &graph.pages {
        if let Some(idx) = graph.index.get(&page.concept) {
            let out_degree = graph.graph.edges(*idx).count() as f64;
            if avg_out > 0.0 && out_degree < (avg_out * 0.5) {
                report.add(format!(
                    "Low link density in {} (outgoing {:.0}, avg {:.1})",
                    page.rel_path, out_degree, avg_out
                ));
            }
        }
    }

    let mut index_nodes = Vec::new();
    for (_concept, idx) in &graph.index {
        if let Some(label) = graph.graph.node_weight(*idx) {
            if label.ends_with("index.md") {
                index_nodes.push(*idx);
            }
        }
    }

    if !index_nodes.is_empty() {
        let mut reachable: HashSet<petgraph::graph::NodeIndex> = HashSet::new();
        for root in index_nodes {
            let mut dfs = Dfs::new(&graph.graph, root);
            while let Some(nx) = dfs.next(&graph.graph) {
                reachable.insert(nx);
            }
        }

        for page in &graph.pages {
            if let Some(idx) = graph.index.get(&page.concept) {
                if !reachable.contains(idx) {
                    report.add(format!("Unreachable page: {}", page.rel_path));
                }
            }
        }
        return;
    }

    let mut referenced: HashSet<String> = HashSet::new();
    for page in &graph.pages {
        for link in &page.links {
            referenced.insert(link.clone());
        }
    }
    for page in &graph.pages {
        if !referenced.contains(&page.concept) && graph.pages.len() > 1 {
            report.add(format!("Orphan page: {}", page.rel_path));
        }
    }
}
