import logging
from app.core.common.neo4j_connection import Neo4jConnection

logger = logging.getLogger(__name__)

def prune_non_wiki_descriptions(conn: Neo4jConnection) -> int | None:
    """
    Sets the description property to null for all Term nodes where 
    wiki_fact_checked is 'No' and the description is not already null.

    Args:
        conn: An active Neo4jConnection instance.

    Returns:
        The number of nodes updated, or None if an error occurred.
    """
    prune_query = """
    MATCH (n:Term)
    WHERE n.wiki_fact_checked = 'No'
    WITH n WHERE n.description IS NOT NULL // Only affect nodes that actually have a description
    SET n.description = null
    RETURN count(n) AS nodes_updated
    """
    check_query = "MATCH (n:Term) WHERE n.wiki_fact_checked = 'No' RETURN count(n) AS potential_nodes"
    
    try:
        results = conn.query(prune_query) # Use query() to get the count back
        if results:
            count = results[0]["nodes_updated"]
            logger.info(f"Utility: Pruned descriptions for {count} node(s) with wiki_fact_checked='No'.")
            return count
        else:
            # Query might have failed or returned no results (e.g., 0 updated)
            # Check if there were potential nodes to update
            check_results = conn.query(check_query)
            potential_count = check_results[0]["potential_nodes"] if check_results else 0
            if potential_count > 0:
                logger.info("Utility: Description pruning: No nodes affected (Description might have been null already).")
            else:
                logger.info("Utility: Description pruning: No nodes found with wiki_fact_checked='No'.")
            return 0 # Return 0 if no nodes were updated

    except Exception as e:
        logger.error(f"Utility: Error during description pruning query: {e}", exc_info=True)
        return None # Indicate error by returning None 