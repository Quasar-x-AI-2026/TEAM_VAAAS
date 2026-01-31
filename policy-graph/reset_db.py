from neo4j import GraphDatabase

# Your Neo4j Credentials
URI = "neo4j+s://11283f72.databases.neo4j.io"
AUTH = ("neo4j", "fUj3kqKVq1rYniF_akSB-5aUq4pql2tg7Ux2vOC5MTg")

def reset():
    driver = GraphDatabase.driver(URI, auth=AUTH)
    try:
        with driver.session() as session:
            print("🗑️  Wiping Neo4j Database...")
            # Deletes all nodes and relationships
            session.run("MATCH (n) DETACH DELETE n")
            print("✅ Database Wiped Clean.")
            print("👉 Now restart 'server.py' to re-ingest the PDFs correctly.")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        driver.close()

if __name__ == "__main__":
    reset()
