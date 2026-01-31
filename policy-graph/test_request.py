import requests
import json

# The URL of your local server
url = "http://localhost:8000/analyze"

# The query you want to run
params = {"query": "How does NEP 2020 impact healthcare education?"}

print("⏳ Sending request to server... (This takes time!)")
response = requests.get(url, params=params)

if response.status_code == 200:
    data = response.json()
    
    # 1. Show Stats
    print("\n📊 STATS:")
    print(f"Impact Factor: {data['stats']['val']}")
    print(f"Message: {data['stats']['msg']}")
    
    # 2. Show Graph Nodes
    print(f"\n🕸️ GRAPH: Found {len(data['graph']['nodes'])} nodes and {len(data['graph']['edges'])} edges.")
    
    # 3. Show Reports
    print("\n📝 REPORTS GENERATED:")
    for report in data['reports']:
        print(f" - {report['role']} (Has Diagram? {'Yes' if report['diagram_code'] else 'No'})")
        
    # Optional: Save to file to inspect
    with open("response.json", "w") as f:
        json.dump(data, f, indent=2)
    print("\n✅ Full JSON saved to 'response.json'")
    
else:
    print(f"❌ Error: {response.text}")