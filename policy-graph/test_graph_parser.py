import re
import json

def parse_mermaid_to_json(mermaid_code: str):
    """
    Parses Mermaid to a simple list of nodes and edges.
    Output format: [ { "group": "nodes", "data": { "id": "A", "label": "Text" } }, ... ]
    """
    if not mermaid_code:
        return []

    elements = []
    seen_ids = set()

    # Regex 1: Matches A[Label] or A["Label"]
    node_pattern = re.compile(r'([a-zA-Z0-9_]+)\s*\["?(.*?)"?\]')
    
    # Regex 2: Matches A --> B, A -.-> B, A ==> B
    edge_pattern = re.compile(r'([a-zA-Z0-9_]+)\s*[-=.].*?>\s*([a-zA-Z0-9_]+)')

    lines = mermaid_code.split('\n')
    
    for line in lines:
        line = line.strip()
        
        # 1. Extract Explicit Nodes
        node_matches = node_pattern.findall(line)
        for node_id, label in node_matches:
            if node_id not in seen_ids:
                clean_label = label.strip().strip('"').replace("<br>", " ")
                elements.append({
                    "group": "nodes",
                    "data": { "id": node_id, "label": clean_label }
                })
                seen_ids.add(node_id)

        # 2. Extract Edges (and implicit nodes)
        edge_matches = edge_pattern.findall(line)
        for source, target in edge_matches:
            if source not in seen_ids:
                elements.append({ "group": "nodes", "data": {"id": source, "label": source} })
                seen_ids.add(source)
            
            if target not in seen_ids:
                elements.append({ "group": "nodes", "data": {"id": target, "label": target} })
                seen_ids.add(target)

            elements.append({
                "group": "edges",
                "data": { "source": source, "target": target }
            })

    return elements

# --- TEST CASES ---
test_cases = [
    {
        "name": "Simple Flow",
        "input": """
        graph TD
        A[Policy] --> B[Impact]
        B --> C[Outcome]
        """
    },
    {
        "name": "Complex Labels",
        "input": """
        graph LR
        N1["User Privacy"] -.-> N2["Data Security"]
        N2 ==> N3[Trust]
        """
    },
    {
        "name": "Implicit Nodes",
        "input": """
        X --> Y
        Y --> Z
        """
    }
]

print("🔍 Running Graph Parser Tests...\n")

for test in test_cases:
    print(f"--- Test: {test['name']} ---")
    output = parse_mermaid_to_json(test['input'])
    print(json.dumps(output, indent=2))
    
    # Validation
    has_nodes = any(e['group'] == 'nodes' for e in output)
    has_edges = any(e['group'] == 'edges' for e in output)
    
    if has_nodes and has_edges:
        print("✅ PASS: Correctly identified nodes and edges.\n")
    else:
        print("❌ FAIL: Missing nodes or edges.\n")
