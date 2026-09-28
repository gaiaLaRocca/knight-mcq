from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, Neo4jError
import logging

logger = logging.getLogger(__name__)

# The edge from the node whose text was mined (P) to the head of a triplet extracted from it.
# It asserts only that the head appears in P's text. Written by `process_triplet` (chatbot.py);
# `find_paths` allows at most one per path. See thesis-concept-robustness/docs/
# phase1_kb_quality_plan.md, "Relation hygiene: parent edges".
ASSOCIATION_RELATION = "ASSOCIATED_WITH"

class Neo4jConnection:
    def __init__(self, uri, user, pwd):
        self._uri = uri
        self._user = user
        self._pwd = pwd
        self._driver = None
        self._connect()

    def _connect(self):
        try:
            self._driver = GraphDatabase.driver(self._uri, auth=(self._user, self._pwd))
            logger.info("Successfully connected to Neo4j.")
        except Exception as e:
            logger.critical(f"Failed to create the driver: {e}")
            raise

    def close(self):
        if self._driver:
            self._driver.close()
            logger.info("Neo4j driver closed.")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def query(self, query, parameters=None, **kwargs):
        if not self._driver:
            logger.warning("Driver not initialized.")
            return None
        try:
            with self._driver.session() as session:
                result = session.run(query, parameters or {}, **kwargs)
                return list(result)
        except Neo4jError as ne:
            logger.error(f"Neo4j error during query execution: {ne}")
        except ServiceUnavailable as su:
            logger.error(f"Service unavailable: {su}")
        except Exception as e:
            logger.error(f"Query failed: {e}")
        return None

    def execute_write(self, query, parameters=None, **kwargs):
        if not self._driver:
            logger.warning("Driver not initialized, cannot execute write.")
            return
        try:
            with self._driver.session() as session:
                session.write_transaction(lambda tx: tx.run(query, parameters or {}, **kwargs))
                logger.info("Write operation executed successfully.")
        except Neo4jError as ne:
            logger.error(f"Neo4j error during write: {ne}")
            raise
        except ServiceUnavailable as su:
            logger.error(f"Service unavailable during write: {su}")
            raise
        except Exception as e:
            logger.error(f"Write execution failed: {e}")
            raise

    def get_all_triples(self):
        """Fetches all (Term)-[REL]->(Term) triples from the graph."""
        logger.info("Fetching all triples from the graph...")
        cypher_query = """
        MATCH (s:Term)-[r]->(o:Term)
        RETURN s.name AS subject, type(r) AS predicate, o.name AS object
        """
        try:
            results = self.query(cypher_query)
            if results is not None:
                triples = [(record["subject"], record["predicate"], record["object"]) for record in results]
                logger.info(f"Successfully fetched {len(triples)} triples.")
                return triples
            else:
                logger.warning("Failed to fetch triples, query returned None.")
                return []
        except Exception as e:
            logger.error(f"Error fetching triples: {e}")
            return []

    def get_all_nodes_with_details(self):
        """Fetches all Term nodes with their name, description, and wiki_fact_checked status."""
        logger.info("Fetching all nodes with details from the graph...")
        cypher_query = """
        MATCH (n:Term)
        RETURN n.name AS name, n.description AS description, n.wiki_fact_checked AS fact_checked
        """
        try:
            results = self.query(cypher_query)
            if results is not None:
                nodes = [
                    {
                        "name": record["name"],
                        "description": record["description"],
                        "fact_checked": record["fact_checked"]
                    }
                    for record in results
                ]
                logger.info(f"Successfully fetched details for {len(nodes)} nodes.")
                return nodes
            else:
                logger.warning("Failed to fetch node details, query returned None.")
                return []
        except Exception as e:
            logger.error(f"Error fetching node details: {e}")
            return []

    def find_paths(self, max_length=2, exact_length=None, start_name=None):
        """Finds paths in the graph.

        If exact_length is specified, finds paths with exactly that many relationships.
        Otherwise, finds paths with 1 to max_length relationships.

        Args:
            max_length (int): Max relationships if exact_length is None.
            exact_length (int | None): Exact number of relationships, or None.
            start_name (str | None): When set, only paths starting at the node with exactly
                this name. Without it, a path may start at any node: on test_7, 23 of the 49
                paths did (`marin county -> san francisco bay area`, `leon moisseiff ->
                golden gate bridge design`...), and 6 of the 21 surviving questions never
                named the bridge, while every path still costs two model calls. See
                thesis-concept-robustness/docs/phase1_kb_quality_plan.md, "Paths enumerated
                from the topic only".

        Returns:
            list: List of path dictionaries, or empty list on error.
        """
        path_length_pattern = ""
        log_length_msg = ""
        query_limit = 1000 # Default safety limit

        if exact_length is not None:
            if exact_length < 1:
                logger.warning("exact_length must be at least 1.")
                return []
            path_length_pattern = f"[*_exact_length_]"
            log_length_msg = f"exactly length {exact_length}"
            # Allow potentially more results when specific length is requested?
            # query_limit = 2000 
        else:
            if max_length < 1:
                 logger.warning("max_length must be at least 1.")
                 return []
            path_length_pattern = f"[*1..{int(max_length)}]"
            log_length_msg = f"up to length {max_length}"
            
        # The start is bound by a parameter, never interpolated: node names are model output.
        start_pattern = "(start_node:Term {name: $start_name})" if start_name else "(start_node:Term)"
        if start_name:
            log_length_msg += f", starting at '{start_name}'"
        logger.info(f"Finding paths with {log_length_msg}...")

        # At most one association hop per path: a path made only of them (topic -> A -> B,
        # neither link stated) leaves the question nothing to be about.
        association_filter = (
            f"AND size([rel IN relationships(p) WHERE type(rel) = '{ASSOCIATION_RELATION}']) <= 1"
        )

        # Construct the query using the determined pattern
        # Note: Parameterizing the variable length part (`[*N]`) directly is complex/not standard.
        # Building the string is common practice here.
        if exact_length is not None:
             # Cypher for exact length
            cypher_query = f"""
            MATCH p={start_pattern}-[r*{exact_length}]->(end_node:Term)
            WHERE start_node <> end_node
            {association_filter}
            RETURN
                [node IN nodes(p) | {{name: node.name, description: node.description, fact_checked: node.wiki_fact_checked}}] AS path_nodes,
                [rel IN relationships(p) | type(rel)] AS path_relationships
            LIMIT {query_limit}
            """
        else:
            # Cypher for max length
            cypher_query = f"""
            MATCH p={start_pattern}-[r*1..{int(max_length)}]->(end_node:Term)
            WHERE start_node <> end_node
            {association_filter}
            RETURN
                [node IN nodes(p) | {{name: node.name, description: node.description, fact_checked: node.wiki_fact_checked}}] AS path_nodes,
                [rel IN relationships(p) | type(rel)] AS path_relationships
            LIMIT {query_limit}
            """
        
        try:
            results = self.query(cypher_query, parameters={"start_name": start_name} if start_name else None)
            if results is not None:
                paths = []
                for record in results:
                    if len(record["path_nodes"]) == len(record["path_relationships"]) + 1:
                        paths.append({
                            "nodes": record["path_nodes"],
                            "relationships": record["path_relationships"]
                        })
                    else:
                        logger.warning(f"Inconsistent path found: {len(record['path_nodes'])} nodes, {len(record['path_relationships'])} rels. Skipping.")
                        
                logger.info(f"Successfully found {len(paths)} paths ({log_length_msg}, Query Limit: {query_limit}).")
                return paths
            else:
                logger.warning(f"Failed to find paths ({log_length_msg}), query returned None.")
                return []
        except Exception as e:
            logger.error(f"Error finding paths ({log_length_msg}): {e}")
            return []
