from neo4j import GraphDatabase

# Using your provided credentials
URI = "neo4j+s://11283f72.databases.neo4j.io"
USER = "neo4j"
PASS = "fUj3kqKVq1rYniF_akSB-5aUq4pql2tg7Ux2vOC5MTg"

def verify_graph():
    driver = GraphDatabase.driver(URI, auth=(USER, PASS))
    with driver.session() as session:
        print("🔍 Checking Neo4j Database Contents...")
        
        # 1. Count Policies
        policies = session.run("MATCH (p:Policy) RETURN count(p) AS count").single()["count"]
        
        # 2. Count Sectors
        sectors = session.run("MATCH (s:Sector) RETURN count(s) AS count").single()["count"]
        
        # 3. Count Relationships
        rels = session.run("MATCH ()-[r:AFFECTS]->() RETURN count(r) AS count").single()["count"]
        
        print(f"\n--- DATABASE SUMMARY ---")
        print(f"✅ Policy Nodes: {policies}")
        print(f"✅ Sector Nodes: {sectors}")
        print(f"✅ 'AFFECTS' Relationships: {rels}")
        
        if rels > 0:
            print("\n--- SAMPLE RELATIONSHIP ---")
            sample = session.run("""
                MATCH (p:Policy)-[:AFFECTS]->(s:Sector) 
                RETURN p.name + ' affects ' + s.name AS text 
                LIMIT 1
            """).single()
            if sample:
                print(f"Found: {sample['text']}")
        else:
            print("\n❌ WARNING: Database is connected but NO relationships were found.")
            print("This means the ingestion ran, but nothing was written to the graph.")

    driver.close()

if __name__ == "__main__":
    verify_graph()