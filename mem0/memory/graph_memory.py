import logging
import json
from copy import deepcopy

from mem0.memory.utils import format_entities

try:
    from langchain_neo4j import Neo4jGraph
except ImportError:
    raise ImportError("langchain_neo4j is not installed. Please install it using pip install langchain-neo4j")

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    raise ImportError("rank_bm25 is not installed. Please install it using pip install rank-bm25")

from mem0.graphs.tools import (
    DELETE_MEMORY_STRUCT_TOOL_GRAPH,
    DELETE_MEMORY_TOOL_GRAPH,
    EXTRACT_ENTITIES_STRUCT_TOOL,
    EXTRACT_ENTITIES_TOOL,
    RELATIONS_STRUCT_TOOL,
    RELATIONS_TOOL,
)
from mem0.graphs.utils import EXTRACT_RELATIONS_PROMPT, get_delete_messages, DELETE_RELATIONS_SYSTEM_PROMPT
from mem0.utils.factory import EmbedderFactory, LlmFactory

logger = logging.getLogger(__name__)


class MemoryGraph:
    def __init__(self, config):
        self.config = config
        self.graph = Neo4jGraph(
            self.config.graph_store.config.url,
            self.config.graph_store.config.username,
            self.config.graph_store.config.password,
            self.config.graph_store.config.database,
            refresh_schema=False,
            driver_config={"notifications_min_severity":"OFF"},
        )
        self.embedding_model = EmbedderFactory.create(
            self.config.embedder.provider, self.config.embedder.config, self.config.vector_store.config
        )

        self.llm_provider = "openai_structured"
        if self.config.llm.provider:
            self.llm_provider = self.config.llm.provider
        if self.config.graph_store.llm:
            self.llm_provider = self.config.graph_store.llm.provider

        self.llm = LlmFactory.create(self.llm_provider, self.config.llm.config)
        self.user_id = None
        self.threshold = 0.7

    def add(self, data, filters):
        """
        Adds data to the graph.

        Args:
            data (str): The data to add to the graph.
            filters (dict): A dictionary containing filters to be applied during the addition.
                            Expected to contain 'user_id' and 'actor_id'.
        """
        # Ensure required filter keys are present, or provide defaults.
        if "user_id" not in filters:
            filters["user_id"] = filters.get("agent_id") or filters.get("run_id") or "default_user"
            logger.warning(f"MemoryGraph.add: 'user_id' was missing in filters, defaulted to {filters['user_id']}")

        if "actor_id" not in filters:
            filters["actor_id"] = filters["user_id"] # Default actor_id to user_id if not present
            logger.info(f"MemoryGraph.add: 'actor_id' was missing in filters, defaulted to user_id: {filters['actor_id']}")
        else:
            # Ensure actor_id from filters is lowercase for consistency
            filters["actor_id"] = str(filters["actor_id"]).lower()
            logger.info(f"MemoryGraph.add: Standardized actor_id to lowercase: {filters['actor_id']}")

        entity_type_map = self._retrieve_nodes_from_data(data, filters)
        to_be_added = self._establish_nodes_relations_from_data(data, filters, entity_type_map)
        search_output = self._search_graph_db(node_list=list(entity_type_map.keys()), filters=filters)
        to_be_deleted = self._get_delete_entities_from_search_output(search_output, data, filters)

        deleted_entities = self._delete_entities(to_be_deleted, filters)
        added_entities = self._add_entities(to_be_added, filters, entity_type_map)

        return {"deleted_entities": deleted_entities, "added_entities": added_entities}

    def add_batch(self, messages_details_list):
        """
        Adds a batch of messages to the graph, aiming to minimize LLM calls
        by processing messages in consolidated batches where possible.

        Args:
            messages_details_list (list[dict]): A list of dictionaries, where each
                dictionary contains:
                - 'content' (str): The text content of the message.
                - 'filters' (dict): Filters associated with this message,
                                    including 'user_id' and a lowercased 'actor_id'.
        """
        logger.info(f"MemoryGraph.add_batch received {len(messages_details_list)} messages.")
        if not messages_details_list:
            return {"deleted_entities": [], "added_entities": []}

        # 1. Consolidate messages
        consolidated_text, actor_segments = self._consolidate_messages_for_llm(messages_details_list)
        logger.debug(f"Consolidated text for batch: {consolidated_text}")
        logger.debug(f"Actor segments for batch: {actor_segments}")

        # 2. Extract entities from the batch
        # This returns a list of dicts: {"name": ..., "type": ..., "actor_id": ...}
        all_entities_with_actors = self._retrieve_nodes_from_data_batch(consolidated_text, actor_segments)
        logger.debug(f"All entities from batch: {all_entities_with_actors}")

        # ---- START: Explicitly ensure speaker nodes are created ----
        # Determine base_user_id first as it's needed for speaker node creation
        # base_user_id = None
        # if messages_details_list:
        #     first_message_filters = messages_details_list[0].get('filters', {})
        #     base_user_id = first_message_filters.get('user_id') or \
        #                    first_message_filters.get('agent_id') or \
        #                    first_message_filters.get('run_id') or \
        #                    'default_batch_user'
        # else:
        #     base_user_id = 'default_empty_batch_user'
        #     logger.warning("messages_details_list is empty for add_batch, cannot effectively determine base_user_id for speaker node creation.")

        # actual_speakers = sorted(list(set(msg_detail['filters'].get('actor_id') for msg_detail in messages_details_list if msg_detail['filters'].get('actor_id'))))
        # logger.info(f"Actual unique speakers found in batch messages: {actual_speakers}")

        # for speaker_name in actual_speakers:
        #     speaker_entity_details_found = False
        #     for entity_info in all_entities_with_actors:
        #         if entity_info.get('name') == speaker_name and entity_info.get('type') == 'person' and entity_info.get('actor_id') == speaker_name:
        #             speaker_entity_details_found = True
        #             break

        #     if speaker_entity_details_found:
        #         logger.info(f"Ensuring graph node exists for speaker: {speaker_name}")
        #         speaker_embedding = self.embedding_model.embed(speaker_name)
        #         cypher_merge_speaker = """
        #         MERGE (p:person {name: $speaker_name, user_id: $user_id, actor_id: $actor_id})
        #         ON CREATE SET p.created = timestamp(), p.mentions = 1, p.embedding = $embedding
        #         ON MATCH SET p.mentions = coalesce(p.mentions, 0) + 1, p.embedding = $embedding
        #         """
        #         # Note: updated embedding on MATCH as well, can be debated if it should only be on CREATE
        #         params_speaker = {
        #             "speaker_name": speaker_name,
        #             "user_id": base_user_id,
        #             "actor_id": speaker_name, # Speaker's node has their own name as actor_id
        #             "embedding": speaker_embedding
        #         }
        #         try:
        #             self.graph.query(cypher_merge_speaker, params=params_speaker)
        #             logger.info(f"Ensured/merged node for speaker: {speaker_name} with actor_id: {speaker_name} and user_id: {base_user_id}")
        #         except Exception as e:
        #             logger.error(f"Error ensuring/merging node for speaker {speaker_name}: {e}")
        #     else:
        #         logger.warning(f"Speaker {speaker_name} was in messages but not extracted as a 'person' entity with self as actor_id in all_entities_with_actors. Node for this speaker might not be created if not part of other relationships.")
        # ---- END: Explicitly ensure speaker nodes are created ----

        if not all_entities_with_actors:
            logger.info("No entities extracted from batch, nothing to add to graph.")
            return {"deleted_entities": [], "added_entities": []}

        # Prepare entity_type_map for _add_entities later.
        # This map should ideally be actor-aware if types can vary per actor for same entity name,
        # but for now, we'll make a global map. If an entity appears multiple times (e.g. by different actors)
        # its type might be overwritten here. This might need refinement if types are actor-specific.
        entity_type_map_batch = {e['name']: e['type'] for e in all_entities_with_actors}

        # 3. Establish relationships from the batch
        # Returns list of dicts: {"source": ..., "relationship": ..., "destination": ...}
        # The entities within these dicts are names, actor context is handled by node creation logic.
        relationships_to_add_batch = self._establish_nodes_relations_from_data_batch(
            consolidated_text, all_entities_with_actors, actor_segments
        )
        logger.debug(f"Relationships to add from batch: {relationships_to_add_batch}")

        # 4. Determine base_filters (e.g., user_id) for subsequent operations.
        # Assuming all messages in a batch share the same primary user_id/run_id.
        # This user_id will be used for creating nodes and for searching.
        base_user_id = None
        if messages_details_list:
            first_message_filters = messages_details_list[0].get('filters', {})
            base_user_id = first_message_filters.get('user_id') or \
                           first_message_filters.get('agent_id') or \
                           first_message_filters.get('run_id') or \
                           'default_batch_user' # Fallback
            if 'user_id' not in first_message_filters:
                 logger.warning(f"'user_id' missing in first message filters of batch, derived: {base_user_id}")
        else:
            base_user_id = 'default_empty_batch_user'
            logger.warning("messages_details_list is empty, cannot determine base_user_id effectively.")

        base_filters_for_search_and_delete = {"user_id": base_user_id}
        # actor_id will be handled specifically within _search_graph_db_batch and _delete_entities

        # 5. Search graph for existing relevant data based on extracted entities
        # all_entities_with_actors already contains actor_id for each entity.
        # _search_graph_db_batch will use this for actor-specific searches.
        comprehensive_search_output = self._search_graph_db_batch(all_entities_with_actors, {"user_id": base_user_id}) # Pass determined base_user_id
        logger.debug(f"Comprehensive search output for batch: {comprehensive_search_output}")

        # 6. Identify entities/relationships to be deleted based on new batched info
        # The `base_user_id` is passed here as a general self-reference for the prompt,
        # but the LLM is guided by actor prefixes in consolidated_text.
        relationships_to_delete_batch = self._get_delete_entities_from_search_output_batch(
            comprehensive_search_output, consolidated_text, actor_segments, base_user_id
        )
        logger.debug(f"Relationships to delete from batch: {relationships_to_delete_batch}")

        # 7. Perform deletions and additions
        # _delete_entities and _add_entities operate on lists and handle actor_id internally
        # based on the 'filters' provided for each operation or within the node data itself.
        # For _delete_entities, it needs to know which actor's relationships to delete.
        # This requires careful handling if `relationships_to_delete_batch` doesn't contain actor info per deletion.
        # Let's assume _delete_entities will be enhanced or it works on global user_id + specific node names for now.
        # The current _delete_entities takes 'filters' which includes an actor_id. We need to iterate deletions per actor.

        final_deleted_results = []
        if relationships_to_delete_batch:
            # Group deletions by the implicit actor context. This is tricky as the LLM output for deletion
            # might not explicitly state actor for each deletion. We need to infer it or improve LLM output.
            # For now, let's assume deletions are associated with the base_user_id and the specific nodes found by LLM.
            # A more robust approach would be to have the LLM specify actor_id for each deletion if possible.
            # Or, iterate through actors and ask for deletions related to them based on their segment of text.
            # This is a simplification for now, applying deletions under the main user_id context.
            # It's assumed the LLM's identified deletions are specific enough (source-rel-dest) that actor ambiguity
            # during delete Cypher is handled by matching nodes that could be actor-specific or general.
            # The existing _delete_entities iterates through items and uses the actor_id from the filters provided to it.
            # We need to ensure each deletion operation gets the correct actor_id context.

            # To correctly use _delete_entities, we need to associate each deletion with its actor_id.
            # The `relationships_to_delete_batch` from LLM contains source, relationship, destination.
            # We need to map these back to actor_ids. This mapping is not straightforward from current LLM output for deletions.
            # This is a significant gap.
            # For now, we will attempt to delete by iterating through unique actor_ids present in the batch
            # and applying all deletions under each actor's context. This is an over-deletion risk if not careful.
            # A safer initial approach: apply deletions with the base_user_id and no specific actor_id in filters,
            # relying on node names being unique or Cypher in _delete_entities handling actor_id if present on nodes.

            # Simplification: Assume _delete_entities can handle actor-specific deletion if nodes have actor_id.
            # Pass a filter with the base_user_id. Actor_id will be determined by Cypher matching.
            # This requires `_delete_entities` to have robust Cypher for matching nodes that might or might not have actor_id.
            # The current `_delete_entities` *does* query based on actor_id from its input filters.
            # So, we must call it per actor for whom deletions are identified.
            # The current `relationships_to_delete_batch` does NOT contain actor_ids. This is a problem.

            # Let's refine: For now, since relationships_to_delete_batch does not have actor_id,
            # we will make a simplifying assumption and pass the base_user_id filter. This means deletions are not actor-specific
            # at the _delete_entities call level, relying on node names + user_id to be specific enough.
            # This is a known limitation of the current batched deletion identification step.
            logger.warning("Batched deletion is applying deletes primarily based on user_id due to LLM output format for deletions. Actor-specific deletion within batch needs refinement.")
            deletions_for_base_user = []
            # Create a temporary filter for deletion. This is NOT ideal.
            # The LLM should ideally return actor_id per deletion.
            # unique_actors_in_batch = list(set(seg['actor_id'] for seg in actor_segments)) # Already have actual_speakers
            for an_actor_id_for_delete_context in actual_speakers: # Iterate actual speakers for delete context
                # actor_specific_delete_filter = base_filters_for_search_and_delete.copy()
                # actor_specific_delete_filter['actor_id'] = an_actor_id_for_delete_context
                # This is still problematic as `relationships_to_delete_batch` applies to ALL actors.
                # We should probably call _get_delete_entities_from_search_output_batch PER ACTOR if we want fine-grained deletion.
                # That defeats the purpose of one LLM call for deletions.

                # Fallback to a simpler (but less accurate) deletion strategy for now:
                # Pass all identified deletions to _delete_entities with a generic filter that has user_id
                # and a primary actor_id from the batch (e.g., the first one). This is NOT robust.
                temp_delete_filters = {"user_id": base_user_id} # base_filters_for_search_and_delete.copy()
                if actual_speakers:
                     temp_delete_filters['actor_id'] = actual_speakers[0] # Use first actor as context for all deletions - BAD!
                else: # Should not happen if messages_details_list is not empty
                     temp_delete_filters['actor_id'] = base_user_id

                deleted_for_this_context = self._delete_entities(relationships_to_delete_batch, temp_delete_filters)
                final_deleted_results.extend(deleted_for_this_context)
                if relationships_to_delete_batch and actual_speakers:
                    logger.warning(f"Applied all batch deletions under context of actor: {temp_delete_filters['actor_id']}. This needs refinement.")
                    break # Avoid multiple deletions of same items under different actor contexts


        final_added_results = []
        if relationships_to_add_batch:
            # _add_entities needs the `entity_type_map` and `filters` containing user_id and actor_id for each node.
            # The `relationships_to_add_batch` gives us {source, relationship, destination}.
            # We need to associate each S-R-D triple with the correct actor context for node creation.
            # The `all_entities_with_actors` list gives us {"name": ..., "type": ..., "actor_id": ...}.
            # We can use this to provide the correct actor_id when calling _add_entities for each relationship.

            # Iterate through relationships to add. For each, determine the actor_id of the source/destination
            # to pass in filters to _add_entities. This assumes _add_entities handles one S-R-D at a time.
            for rel_to_add in relationships_to_add_batch:
                source_name = rel_to_add['source']
                # Find actor_id for source entity. This assumes entity names are unique identifiers within the batch context
                # or that the first match is sufficient. If an entity like "meeting" is mentioned by two actors,
                # which actor_id does it get? `all_entities_with_actors` might have it listed twice with different actor_ids.
                # The LLM for relationship extraction should ideally disambiguate or we need a strategy.
                # For now, take the first actor_id found for an entity name.
                source_actor_id = base_user_id # Default
                dest_actor_id = base_user_id # Default

                found_source_actor = False
                for entity_detail in all_entities_with_actors:
                    if entity_detail['name'] == source_name:
                        source_actor_id = entity_detail['actor_id']
                        found_source_actor = True
                        break
                if not found_source_actor:
                    logger.warning(f"Could not find actor_id for source entity '{source_name}' in all_entities_with_actors. Defaulting.")

                # The `_add_entities` function takes a list of relationships.
                # And it uses one set of `filters` for all of them. This needs to change for batch.
                # `_add_entities` logic needs to be called per relationship with specific filters for that relationship's actors.
                # This is a significant refactoring of _add_entities or this loop.

                # Quick Fix: Let _add_entities get actor_id from the node properties if they exist, or use filters.
                # The current _add_entities MERGEs nodes using actor_id from filters. This is fine if we pass it.
                # We need to call _add_entities for each S-R-D with the specific actor_id(s) of the source/destination.
                # Let's assume for now relationship involves entities from the SAME actor primarily, or the source actor dictates context.
                # This is a simplification.

                add_filters = {"user_id": base_user_id} # base_filters_for_search_and_delete.copy()
                add_filters['actor_id'] = source_actor_id # Use source_actor_id as the context for adding this relationship.

                # _add_entities expects a list of relations, and entity_type_map_batch
                added_rels = self._add_entities([rel_to_add], add_filters, entity_type_map_batch)
                final_added_results.extend(added_rels)

        logger.info(f"MemoryGraph.add_batch finished. Added: {len(final_added_results)}, Deleted: {len(final_deleted_results)}")
        return {"deleted_entities": final_deleted_results, "added_entities": final_added_results}

    def _consolidate_messages_for_llm(self, messages_details_list):
        """
        Consolidates multiple messages into a single string for LLM processing,
        prefixing each message with its actor_id and adding separators.
        Also returns a structure to map parts of the string to original actors.

        Args:
            messages_details_list (list[dict]): List of message details,
                each with 'content' and 'filters' (containing 'actor_id').

        Returns:
            tuple: (consolidated_text, actor_segments)
                - consolidated_text (str): A single string with all messages,
                  formatted with actor prefixes and separators.
                - actor_segments (list[dict]): A list of dicts, each detailing
                  an actor segment in the consolidated text, e.g.,
                  {'actor_id': 'claude', 'start_char': 0, 'end_char': 20, 'original_message_index': 0}
        """
        consolidated_parts = []
        actor_segments = []
        current_char_offset = 0
        separator = "\n---\n" # Define a clear separator

        for index, msg_detail in enumerate(messages_details_list):
            content = msg_detail['content']
            # actor_id should already be lowercased by the calling Memory._add_to_graph logic
            actor_id = msg_detail['filters'].get('actor_id', 'unknown_actor')

            prefix = f"{actor_id}: "
            formatted_message = prefix + content

            consolidated_parts.append(formatted_message)

            segment_start_char = current_char_offset
            segment_end_char = current_char_offset + len(formatted_message)
            actor_segments.append({
                'actor_id': actor_id,
                'start_char': segment_start_char,
                'end_char': segment_end_char,
                'original_message_index': index
            })

            # Update offset for the next message, including the separator length if it's not the last message
            current_char_offset = segment_end_char
            if index < len(messages_details_list) - 1:
                current_char_offset += len(separator)

        consolidated_text = separator.join(consolidated_parts)
        return consolidated_text, actor_segments

    def search(self, query, filters, limit=100):
        """
        Search for memories and related graph data.

        Args:
            query (str): Query to search for.
            filters (dict): A dictionary containing filters to be applied during the search.
            limit (int): The maximum number of nodes and relationships to retrieve. Defaults to 100.

        Returns:
            dict: A dictionary containing:
                - "contexts": List of search results from the base data store.
                - "entities": List of related graph data based on the query.
        """
        entity_type_map = self._retrieve_nodes_from_data(query, filters)
        search_output = self._search_graph_db(node_list=list(entity_type_map.keys()), filters=filters)

        if not search_output:
            return []

        search_outputs_sequence = [
            [item["source"], item["relationship"], item["destination"]] for item in search_output
        ]
        bm25 = BM25Okapi(search_outputs_sequence)

        tokenized_query = query.split(" ")
        reranked_results = bm25.get_top_n(tokenized_query, search_outputs_sequence, n=5)

        search_results = []
        for item in reranked_results:
            search_results.append({"source": item[0], "relationship": item[1], "destination": item[2]})

        logger.info(f"Returned {len(search_results)} search results")

        return search_results

    def delete_all(self, filters):
        cypher = """
        MATCH (n {user_id: $user_id})
        DETACH DELETE n
        """
        params = {"user_id": filters["user_id"]}
        self.graph.query(cypher, params=params)

    def get_all(self, filters, limit=100):
        """
        Retrieves all nodes and relationships from the graph database based on optional filtering criteria.

        Args:
            filters (dict): A dictionary containing filters to be applied during the retrieval.
            limit (int): The maximum number of nodes and relationships to retrieve. Defaults to 100.
        Returns:
            list: A list of dictionaries, each containing:
                - 'contexts': The base data store response for each memory.
                - 'entities': A list of strings representing the nodes and relationships
        """

        # return all nodes and relationships
        query = """
        MATCH (n {user_id: $user_id})-[r]->(m {user_id: $user_id})
        RETURN n.name AS source, type(r) AS relationship, m.name AS target
        LIMIT $limit
        """
        results = self.graph.query(query, params={"user_id": filters["user_id"], "limit": limit})

        final_results = []
        for result in results:
            final_results.append(
                {
                    "source": result["source"],
                    "relationship": result["relationship"],
                    "target": result["target"],
                }
            )

        logger.info(f"Retrieved {len(final_results)} relationships")

        return final_results

    def _retrieve_nodes_from_data(self, data, filters):
        """Extracts all the entities mentioned in the query."""
        _tools = [EXTRACT_ENTITIES_TOOL]
        if self.llm_provider in ["azure_openai_structured", "openai_structured"]:
            _tools = [EXTRACT_ENTITIES_STRUCT_TOOL]

        # Use actor_id for self-reference if available, otherwise fall back to user_id
        self_reference_id = filters.get("actor_id") or filters.get("user_id") or "user"

        search_results = self.llm.generate_response(
            messages=[
                {
                    "role": "system",
                    "content": f"You are a smart assistant who understands entities and their types in a given text. If user message contains self reference such as 'I', 'me', 'my' etc. then use {self_reference_id} as the source entity. Extract all the entities from the text. ***DO NOT*** answer the question itself if the given text is a question.",
                },
                {"role": "user", "content": data},
            ],
            tools=_tools,
        )

        entity_type_map = {}

        try:
            for tool_call in search_results["tool_calls"]:
                if tool_call["name"] != "extract_entities":
                    continue
                for item in tool_call["arguments"]["entities"]:
                    entity_type_map[item["entity"]] = item["entity_type"]
        except Exception as e:
            logger.exception(
                f"Error in search tool: {e}, llm_provider={self.llm_provider}, search_results={search_results}"
            )

        entity_type_map = {k.lower().replace(" ", "_"): v.lower().replace(" ", "_") for k, v in entity_type_map.items()}
        logger.debug(f"Entity type map: {entity_type_map}\n search_results={search_results}")
        return entity_type_map

    def _establish_nodes_relations_from_data(self, data, filters, entity_type_map):
        """Eshtablish relations among the extracted nodes."""

        # Use actor_id for context in prompt if available, otherwise fall back to user_id
        context_id_for_prompt = filters.get("actor_id") or filters.get("user_id") or "user"

        if self.config.graph_store.custom_prompt:
            messages = [
                {
                    "role": "system",
                    "content": EXTRACT_RELATIONS_PROMPT.replace("USER_ID", context_id_for_prompt).replace(
                        "CUSTOM_PROMPT", f"4. {self.config.graph_store.custom_prompt}"
                    ),
                },
                {"role": "user", "content": data},
            ]
        else:
            messages = [
                {
                    "role": "system",
                    "content": EXTRACT_RELATIONS_PROMPT.replace("USER_ID", context_id_for_prompt),
                },
                {"role": "user", "content": f"List of entities: {list(entity_type_map.keys())}. \n\nText: {data}"},
            ]

        _tools = [RELATIONS_TOOL]
        if self.llm_provider in ["azure_openai_structured", "openai_structured"]:
            _tools = [RELATIONS_STRUCT_TOOL]

        extracted_entities = self.llm.generate_response(
            messages=messages,
            tools=_tools,
        )

        entities = []
        if extracted_entities["tool_calls"]:
            entities = extracted_entities["tool_calls"][0]["arguments"]["entities"]

        entities = self._remove_spaces_from_entities(entities)
        logger.debug(f"Extracted entities: {entities}")
        return entities

    def _search_graph_db(self, node_list, filters, limit=100):
        """Search similar nodes among and their respective incoming and outgoing relations."""
        result_relations = []
        user_id = filters["user_id"]
        actor_id = filters.get("actor_id", user_id)

        for node in node_list:
            n_embedding = self.embedding_model.embed(node)

            cypher_query = """
            MATCH (n)
            WHERE n.embedding IS NOT NULL
              AND n.user_id = $user_id
              AND (n.actor_id = $actor_id OR n.actor_id IS NULL)
            WITH n, round(2 * vector.similarity.cosine(n.embedding, $n_embedding) - 1, 4) AS similarity
            WHERE similarity >= $threshold
            CALL {
                WITH n, similarity
                MATCH (n)-[r]->(m)
                WHERE (m.user_id = $user_id AND (m.actor_id = $actor_id OR m.actor_id IS NULL))
                RETURN n.name AS source, elementId(n) AS source_id, type(r) AS relationship, elementId(r) AS relation_id, m.name AS destination, elementId(m) AS destination_id, similarity AS current_similarity, elementId(n) AS n_elementId, n.actor_id AS n_actor_id
                UNION
                WITH n, similarity
                MATCH (m)-[r]->(n)
                WHERE (m.user_id = $user_id AND (m.actor_id = $actor_id OR m.actor_id IS NULL))
                RETURN m.name AS source, elementId(m) AS source_id, type(r) AS relationship, elementId(r) AS relation_id, n.name AS destination, elementId(n) AS destination_id, similarity AS current_similarity, elementId(n) AS n_elementId, n.actor_id AS n_actor_id
            }
            RETURN distinct source, source_id, relationship, relation_id, destination, destination_id, current_similarity, n_elementId, n_actor_id
            ORDER BY current_similarity DESC,
                     CASE WHEN source_id = n_elementId AND n_actor_id = $actor_id THEN 0 ELSE 1 END,
                     CASE WHEN destination_id = n_elementId AND n_actor_id = $actor_id THEN 0 ELSE 1 END
            LIMIT $limit
            """
            params = {
                "n_embedding": n_embedding,
                "threshold": self.threshold,
                "user_id": user_id,
                "actor_id": actor_id,
                "limit": limit,
            }

            try:
                relations = self.graph.query(cypher_query, params=params)
                logging.info(f"relations for {actor_id}: {relations}")
                if relations:
                    for rel in relations:
                        # Ensure all required keys are present before adding
                        if all(k in rel for k in ["source", "relationship", "destination", "current_similarity"]):
                            result_relations.append({
                                "source": rel["source"],
                                "relationship": rel["relationship"],
                                "destination": rel["destination"],
                                "similarity": rel["current_similarity"]
                            })
                        else:
                            logger.warning(f"Skipping relation due to missing keys: {rel}")
            except Exception as e:
                logger.error(f"Error executing graph search query for node '{node}': {e}")

        # Deduplicate final results if needed, though DISTINCT in Cypher helps
        # final_results = [dict(t) for t in {tuple(d.items()) for d in result_relations}]
        # Sort final result_relations by similarity one last time, as they are aggregated from multiple node searches
        # However, the primary sorting happens within Cypher per queried node.
        # If a global re-sort is needed: result_relations.sort(key=lambda x: x.get('similarity', 0), reverse=True)
        return result_relations

    def _get_delete_entities_from_search_output(self, search_output, data, filters):
        """Get the entities to be deleted from the search output."""
        search_output_string = format_entities(search_output)

        # Use actor_id for self-reference in deletion context, fallback to user_id
        self_reference_id_for_delete = filters.get("actor_id") or filters.get("user_id") or "user"

        system_prompt, user_prompt = get_delete_messages(search_output_string, data, self_reference_id_for_delete)


        _tools = [DELETE_MEMORY_TOOL_GRAPH]
        if self.llm_provider in ["azure_openai_structured", "openai_structured"]:
            _tools = [
                DELETE_MEMORY_STRUCT_TOOL_GRAPH,
            ]

        memory_updates = self.llm.generate_response(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            tools=_tools,
        )

        logging.info(f"memory_updates: {memory_updates}")

        to_be_deleted = []
        for item in memory_updates["tool_calls"]:
            if item["name"] == "delete_graph_memory":
                to_be_deleted.append(item["arguments"])
        # in case if it is not in the correct format
        to_be_deleted = self._remove_spaces_from_entities(to_be_deleted)
        logger.debug(f"Deleted relationships: {to_be_deleted}")
        return to_be_deleted

    def _delete_entities(self, to_be_deleted, filters):
        """Delete the entities from the graph, considering user_id and actor_id."""
        logging.info(f"to_be_deleted: {to_be_deleted} -- {filters}")
        results = []
        user_id = filters["user_id"]
        # actor_id from filters is no longer used directly in the MATCH clause for deletion of specific S-R-D.
        # Deletion is now based on source name, destination name, relationship type, and user_id.
        # This makes it more robust if the LLM identifies a semantic deletion regardless of how actor_ids are stored on nodes.

        for item in to_be_deleted:
            source = item["source"]
            destination = item["destination"]
            relationship = item["relationship"]

            # Cypher query to find and delete the specific relationship.
            # It captures the relationship type before deletion for logging/confirmation.
            cypher_capture_then_delete = f"""
            MATCH (n {{name: $source_name, user_id: $user_id}})
            MATCH (m {{name: $dest_name, user_id: $user_id}})
            MATCH (n)-[r_to_delete:{relationship}]->(m)
            WITH n, m, r_to_delete, type(r_to_delete) as relationship_type
            DELETE r_to_delete
            RETURN n.name AS source, m.name AS target, relationship_type AS relationship
            """

            params = {
                "source_name": source,
                "dest_name": destination,
                "user_id": user_id,
            }
            try:
                result = self.graph.query(cypher_capture_then_delete, params=params)
                logging.info(f"Attempted delete for {source}-{relationship}-{destination} under user_id {user_id}. Result: {result}")
                if result: # If the query found and deleted something, result will not be empty.
                    # result is usually a list of dicts, e.g., [{'source': 'bob', 'target': 'alice', 'relationship': 'is_friend_of'}]
                    results.append({"source": source, "relationship": relationship, "destination": destination, "status": "deleted", "details": result})
                else:
                    logging.info(f"No relationship matched for deletion via _delete_entities: {source}-{relationship}-{destination} under user_id {user_id}")

            except Exception as e:
                # Log the specific S-R-D and user_id for which deletion failed.
                logger.error(f"Error deleting relationship {source}-{relationship}-{destination} for user_id {user_id}: {e}")
        return results

    def _add_entities(self, to_be_added, filters, entity_type_map):
        """Add the new entities to the graph. Merge the nodes if they already exist."""
        results = []
        user_id = filters["user_id"] # Extract user_id from filters
        actor_id = filters.get("actor_id", user_id) # Extract actor_id, default to user_id if not present

        for item in to_be_added:
            # entities
            source = item["source"]
            destination = item["destination"]
            relationship = item["relationship"]

            # types
            source_type = entity_type_map.get(source, "__User__")
            destination_type = entity_type_map.get(destination, "__User__")

            # embeddings
            source_embedding = self.embedding_model.embed(source)
            dest_embedding = self.embedding_model.embed(destination)

            # search for the nodes with the closest embeddings
            # Pass user_id and potentially actor_id if search needs to be actor-aware
            source_node_search_result = self._search_source_node(source_embedding, user_id, actor_id, threshold=0.9)
            destination_node_search_result = self._search_destination_node(dest_embedding, user_id, actor_id, threshold=0.9)

            # TODO: Create a cypher query and common params for all the cases
            if not destination_node_search_result and source_node_search_result:
                cypher = f"""
                    MATCH (source)
                    WHERE elementId(source) = $source_id
                    SET source.mentions = coalesce(source.mentions, 0) + 1
                    WITH source
                    MERGE (destination:{destination_type} {{name: $destination_name, user_id: $user_id, actor_id: $actor_id}})
                    ON CREATE SET
                        destination.created = timestamp(),
                        destination.mentions = 1
                    ON MATCH SET
                        destination.mentions = coalesce(destination.mentions, 0) + 1
                    WITH source, destination
                    CALL db.create.setNodeVectorProperty(destination, 'embedding', $destination_embedding)
                    WITH source, destination
                    MERGE (source)-[r:{relationship}]->(destination)
                    ON CREATE SET
                        r.created = timestamp(),
                        r.mentions = 1
                    ON MATCH SET
                        r.mentions = coalesce(r.mentions, 0) + 1
                    RETURN source.name AS source, type(r) AS relationship, destination.name AS target
                    """

                params = {
                    "source_id": source_node_search_result[0]["elementId(source_candidate)"],
                    "destination_name": destination,
                    "destination_embedding": dest_embedding,
                    "user_id": user_id,
                    "actor_id": actor_id,
                }
            elif destination_node_search_result and not source_node_search_result:
                cypher = f"""
                    MATCH (destination)
                    WHERE elementId(destination) = $destination_id
                    SET destination.mentions = coalesce(destination.mentions, 0) + 1
                    WITH destination
                    MERGE (source:{source_type} {{name: $source_name, user_id: $user_id, actor_id: $actor_id}})
                    ON CREATE SET
                        source.created = timestamp(),
                        source.mentions = 1
                    ON MATCH SET
                        source.mentions = coalesce(source.mentions, 0) + 1
                    WITH source, destination
                    CALL db.create.setNodeVectorProperty(source, 'embedding', $source_embedding)
                    WITH source, destination
                    MERGE (source)-[r:{relationship}]->(destination)
                    ON CREATE SET
                        r.created = timestamp(),
                        r.mentions = 1
                    ON MATCH SET
                        r.mentions = coalesce(r.mentions, 0) + 1
                    RETURN source.name AS source, type(r) AS relationship, destination.name AS target
                    """

                params = {
                    "destination_id": destination_node_search_result[0]["elementId(destination_candidate)"],
                    "source_name": source,
                    "source_embedding": source_embedding,
                    "user_id": user_id,
                    "actor_id": actor_id,
                }
            elif source_node_search_result and destination_node_search_result:
                # Existing nodes are matched by elementId, user_id and actor_id might not need to be in MATCH
                # but good to have in params for consistency if queries evolve.
                # If we update actor_id on existing nodes, that's a different logic.
                # For now, we assume actor_id is primarily for new node creation.
                cypher = f"""
                    MATCH (source)
                    WHERE elementId(source) = $source_id
                    SET source.mentions = coalesce(source.mentions, 0) + 1
                    WITH source
                    MATCH (destination)
                    WHERE elementId(destination) = $destination_id
                    SET destination.mentions = coalesce(destination.mentions) + 1
                    MERGE (source)-[r:{relationship}]->(destination)
                    ON CREATE SET
                        r.created_at = timestamp(),
                        r.updated_at = timestamp(),
                        r.mentions = 1
                    ON MATCH SET r.mentions = coalesce(r.mentions, 0) + 1


                    RETURN source.name AS source, type(r) AS relationship, destination.name AS target
                    """
                params = {
                    "source_id": source_node_search_result[0]["elementId(source_candidate)"],
                    "destination_id": destination_node_search_result[0]["elementId(destination_candidate)"],
                    "user_id": user_id, # Still useful for context, though not in MATCH elementId
                    "actor_id": actor_id, # Same as user_id
                }
            else: # Both source and destination are new or not found via embedding search with sufficient similarity
                cypher = f"""
                    MERGE (n:{source_type} {{name: $source_name, user_id: $user_id, actor_id: $actor_id}})
                    ON CREATE SET n.created = timestamp(),
                                  n.mentions = 1
                    ON MATCH SET n.mentions = coalesce(n.mentions, 0) + 1
                                 // Potentially update actor_id if node exists but actor is different? Or create new node for new actor?
                                 // For now, MERGE on name, user_id, actor_id ensures distinct nodes per actor for same name.
                    WITH n
                    CALL db.create.setNodeVectorProperty(n, 'embedding', $source_embedding)
                    WITH n
                    MERGE (m:{destination_type} {{name: $dest_name, user_id: $user_id, actor_id: $actor_id}})
                    ON CREATE SET m.created = timestamp(),
                                  m.mentions = 1
                    ON MATCH SET m.mentions = coalesce(m.mentions, 0) + 1
                    WITH n, m
                    CALL db.create.setNodeVectorProperty(m, 'embedding', $dest_embedding)
                    WITH n, m
                    MERGE (n)-[rel:{relationship}]->(m)
                    ON CREATE SET rel.created = timestamp(), rel.mentions = 1
                    ON MATCH SET rel.mentions = coalesce(rel.mentions, 0) + 1
                    RETURN n.name AS source, type(rel) AS relationship, m.name AS target
                    """
                params = {
                    "source_name": source,
                    "dest_name": destination,
                    "source_embedding": source_embedding,
                    "dest_embedding": dest_embedding,
                    "user_id": user_id,
                    "actor_id": actor_id,
                }
            result = self.graph.query(cypher, params=params)
            results.append(result)
        return results

    def _remove_spaces_from_entities(self, entity_list):
        for item in entity_list:
            item["source"] = item["source"].lower().replace(" ", "_")
            item["relationship"] = item["relationship"].lower().replace(" ", "_")
            item["destination"] = item["destination"].lower().replace(" ", "_")
        return entity_list

    def _search_source_node(self, source_embedding, user_id, actor_id, threshold=0.9):
        cypher = """
            MATCH (source_candidate)
            WHERE source_candidate.embedding IS NOT NULL
            AND source_candidate.user_id = $user_id
            AND (source_candidate.actor_id = $actor_id OR source_candidate.actor_id IS NULL) // Prefer actor-specific, or general
            WITH source_candidate,
            round(2 * vector.similarity.cosine(source_candidate.embedding, $source_embedding) - 1, 4) AS source_similarity // denormalize for backward compatibility
            WHERE source_similarity >= $threshold

            WITH source_candidate, source_similarity
            ORDER BY source_similarity DESC, CASE WHEN source_candidate.actor_id = $actor_id THEN 0 ELSE 1 END // Prioritize exact actor match
            LIMIT 1

            RETURN elementId(source_candidate)
            """

        params = {
            "source_embedding": source_embedding,
            "user_id": user_id,
            "actor_id": actor_id,
            "threshold": threshold,
        }

        result = self.graph.query(cypher, params=params)
        return result

    def _search_destination_node(self, destination_embedding, user_id, actor_id, threshold=0.9):
        cypher = """
            MATCH (destination_candidate)
            WHERE destination_candidate.embedding IS NOT NULL
            AND destination_candidate.user_id = $user_id
            AND (destination_candidate.actor_id = $actor_id OR destination_candidate.actor_id IS NULL) // Prefer actor-specific, or general
            WITH destination_candidate,
            round(2 * vector.similarity.cosine(destination_candidate.embedding, $destination_embedding) - 1, 4) AS destination_similarity // denormalize for backward compatibility

            WHERE destination_similarity >= $threshold

            WITH destination_candidate, destination_similarity
            ORDER BY destination_similarity DESC, CASE WHEN destination_candidate.actor_id = $actor_id THEN 0 ELSE 1 END // Prioritize exact actor match
            LIMIT 1

            RETURN elementId(destination_candidate)
            """
        params = {
            "destination_embedding": destination_embedding,
            "user_id": user_id,
            "actor_id": actor_id,
            "threshold": threshold,
        }

        result = self.graph.query(cypher, params=params)
        return result

    def _retrieve_nodes_from_data_batch(self, consolidated_text, actor_segments):
        """
        Extracts entities from a consolidated batch of messages using one LLM call.
        Associates extracted entities with their original actors.

        Args:
            consolidated_text (str): The single string of all messages, with actor prefixes.
            actor_segments (list[dict]): Information about actor segments in the text.

        Returns:
            list[dict]: A list of entity information dictionaries, e.g.,
                        [{"name": "claude", "type": "person", "actor_id": "claude"}, ...]
        """
        # TODO: The self.llm_provider check and tool selection should be harmonized
        # if it's consistently used across methods.
        _tools = [EXTRACT_ENTITIES_TOOL]
        if self.llm_provider in ["azure_openai_structured", "openai_structured"]:
            # This tool definition might need to be updated to include 'actor_id' in its output schema
            # For now, we rely on the prompt to guide the LLM.
            _tools = [EXTRACT_ENTITIES_STRUCT_TOOL]

        # Construct a more detailed system prompt for batch processing
        # The actor_segments can be passed to the LLM as part of the context if helpful,
        # or used by our code to map LLM responses back.
        # For now, the prompt primarily guides the LLM based on the inline actor prefixes.

        actor_names_in_batch = sorted(list(set([seg['actor_id'] for seg in actor_segments])))
        actors_string = ", ".join(actor_names_in_batch)

        system_prompt = (
            f"You are a smart assistant specializing in entity extraction from conversations involving multiple actors: {actors_string}.\n"
            f"The input text contains messages prefixed by the speaker's name (e.g., 'actor_name: message content'). "
            f"IMPORTANT: For each unique speaker (actor) in the conversation (e.g., {actors_string}), you MUST extract an entity representing that speaker. "
            f"For example, if 'Alex', and 'Chloe' are speakers, ensure you output entities like: "
            f"  {{'entity': 'alex', 'entity_type': 'person', 'actor_id': 'alex'}}, "
            f"  {{'entity': 'chloe', 'entity_type': 'person', 'actor_id': 'chloe'}}. "
            f"This is in ADDITION to other entities found in their statements. "
            f"Then, extract all other relevant entities from the entire text. "
            f"The 'entity' name field in your output should be the pure name of the entity (e.g., 'fido', 'alice', 'mem0', 'alex'). "
            f"DO NOT include actor information like '(actor: name)' or similar suffixes within the 'entity' name string itself. "
            f"When an entity is a self-reference (e.g., 'I', 'me', 'my'), the 'entity' name MUST be the speaker's name for that part of the text. "
            f"For each extracted entity (including the speaker entities themselves), you MUST populate the separate 'actor_id' field with the speaker's name (e.g., 'claude', 'leo') who mentioned or is the primary subject of that entity in that context. The actor_id for a speaker entity should be their own name. "
            f"Your output must conform to the 'extract_entities' tool schema, which includes an 'actor_id' field per entity. "
            f"Example of topical entity: If 'claude: I like apples.', output: {{'entity': 'apples', 'entity_type': 'fruit', 'actor_id': 'claude'}}. "
            f"***DO NOT*** answer any questions found in the text itself; only extract entities."
        )

        # The LLM call itself
        search_results = self.llm.generate_response(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": consolidated_text},
            ],
            tools=_tools,
        )

        logger.info(f"MemoryGraph.add_batch - all_entities_with_actors: {search_results}") # DETAILED LOGGING


        entity_info_list = []
        # TODO: Robust parsing of search_results["tool_calls"]
        # This parsing logic needs to be adapted based on how the LLM structures its output,
        # especially regarding actor_id. If the tool schema is updated to include actor_id,
        # this becomes more straightforward. If not, we might need more complex logic or
        # rely on the LLM to make entity names actor-specific for self-references.

        try:
            for tool_call in search_results.get("tool_calls", []):
                if tool_call["name"] == "extract_entities":
                    for item in tool_call["arguments"].get("entities", []):
                        entity_name = str(item["entity"]).lower().replace(" ", "_")
                        entity_type = str(item["entity_type"]).lower().replace(" ", "_")

                        # Attempt to get actor_id if provided by the LLM/tool
                        # This is speculative as EXTRACT_ENTITIES_TOOL might not have this field yet.
                        extracted_actor_id = item.get("actor_id")

                        # TODO: If extracted_actor_id is None, we need a robust way to map this entity
                        # back to its segment in consolidated_text and then to its actor via actor_segments.
                        # This could involve approximate string matching or more sophisticated context mapping.
                        # For a first pass, if not directly provided, we might have to leave it or make a best guess.
                        # For now, we will prioritize explicit actor_id if present.
                        # A simple fallback: if entity_name is one of the actor_names_in_batch, assume it's that actor.
                        final_actor_id = extracted_actor_id.lower() if extracted_actor_id else None
                        if not final_actor_id and entity_name in actor_names_in_batch:
                            final_actor_id = entity_name

                        # Fallback if no actor_id can be determined (should be improved)
                        if not final_actor_id:
                             # This is a simplification; true actor attribution would require more complex logic
                             # or a more capable LLM/tool output.
                             # For now, we log a warning. Ideally, every entity should be tied to an actor.
                             logger.warning(f"Could not reliably determine actor_id for entity: {entity_name}. Attributing to 'unknown_actor_batch'.")
                             final_actor_id = 'unknown_actor_batch'

                        entity_info_list.append({
                            "name": entity_name,
                            "type": entity_type,
                            "actor_id": final_actor_id
                        })
        except Exception as e:
            logger.exception(
                f"Error in parsing batched entity extraction results: {e}, llm_provider={self.llm_provider}, search_results={search_results}"
            )

        logger.debug(f"Batched entity_info_list: {entity_info_list}")
        return entity_info_list

    def _establish_nodes_relations_from_data_batch(self, consolidated_text, entity_info_list, actor_segments):
        """
        Establishes relationships from a consolidated batch of messages using one LLM call.
        Ensures relationships respect actor contexts.

        Args:
            consolidated_text (str): The single string of all messages, with actor prefixes.
            entity_info_list (list[dict]): List of extracted entity information,
                                         each like {"name": ..., "type": ..., "actor_id": ...}.
            actor_segments (list[dict]): Information about actor segments in the text.

        Returns:
            list[dict]: A list of relationship dictionaries, e.g.,
                        [{"source": "claude", "relationship": "has_age", "destination": "42_ans"}, ...]
                        (Actor context is implicitly carried by the actor_id of the source/destination nodes)
        """
        if not entity_info_list:
            logger.info("No entities found, skipping relationship extraction.")
            return []

        # Prepare a string representation of entities with their actors for the prompt
        # E.g., "claude (actor: claude), 42_ans (actor: claude), leo (actor: leo)"
        entities_with_actors_str_parts = []
        for entity_info in entity_info_list:
            entities_with_actors_str_parts.append(f"{entity_info['name']} (actor: {entity_info['actor_id']})")
        entities_for_prompt = ", ".join(entities_with_actors_str_parts)

        actor_names_in_batch = sorted(list(set([seg['actor_id'] for seg in actor_segments])))
        actors_string = ", ".join(actor_names_in_batch)

        # System prompt for batched relationship extraction
        # This prompt needs to be carefully designed to handle multi-actor context.
        system_prompt = (
            f"You are an advanced algorithm specializing in extracting relationships for a knowledge graph from conversations involving multiple actors: {actors_string}. "
            f"The input text contains messages prefixed by the speaker\'s name. "
            f"A list of pre-extracted entities, along with their identified actors, is provided: [{entities_for_prompt}]. "
            f"Your task is to establish relationships ONLY between these provided entities based on the entire input text. "
            f"When forming relationships: "
            f"  - Consider the context of who said what. Self-references (like 'I said...' or 'my age is...') should link to the entity representing the speaker of that segment. "
            f"  - Relationships should primarily be between entities associated with the same actor or where the text clearly links entities across different actors. "
            f"  - Use consistent, general, and timeless relationship types (e.g., prefer 'is_friend_of' over 'became_friends_with'). "
            f"Your output must conform to the 'establish_relationships' tool schema, providing source, relationship, and destination for each. "
            f"Focus on explicitly stated information. Do not infer relationships not present in the text."
        )

        if self.config.graph_store.custom_prompt: # Adapting for existing custom prompt logic
            # This might need review if custom_prompt assumes single-actor context
            system_prompt += f"\nAdditionally, follow this custom rule: {self.config.graph_store.custom_prompt}"

        _tools = [RELATIONS_TOOL]
        if self.llm_provider in ["azure_openai_structured", "openai_structured"]:
            _tools = [RELATIONS_STRUCT_TOOL]

        # LLM call
        extracted_relations_response = self.llm.generate_response(
            messages=[
                {"role": "system", "content": system_prompt},
                # Providing entities again in user message might be redundant if already in system, but can reinforce
                {"role": "user", "content": f"Entities list: [{entities_for_prompt}]\n\nFull Text:\n{consolidated_text}"},
            ],
            tools=_tools,
        )

        relationships_to_add = []
        try:
            if extracted_relations_response.get("tool_calls"):
                for tool_call in extracted_relations_response["tool_calls"]:
                    if tool_call["name"] == "establish_relationships":
                        # Ensure arguments are correctly parsed (OpenAI returns string, others might return dict)
                        args = tool_call["arguments"]
                        if isinstance(args, str):
                            try:
                                args = json.loads(args) # Attempt to parse if it's a JSON string
                            except json.JSONDecodeError:
                                logger.error(f"Failed to parse string arguments for establish_relationships: {args}")
                                continue

                        raw_entities = args.get("entities", [])
                        # The _remove_spaces_from_entities helper expects a list of dicts
                        # with 'source', 'relationship', 'destination' keys.
                        # This tool returns entities with these keys directly.
                        relationships_to_add.extend(self._remove_spaces_from_entities(deepcopy(raw_entities)))
        except Exception as e:
            logger.exception(
                f"Error in parsing batched relationship extraction results: {e}, response: {extracted_relations_response}"
            )

        logger.debug(f"Batched relationships_to_add: {relationships_to_add}")
        return relationships_to_add

    def _search_graph_db_batch(self, entity_info_list, base_filters):
        """
        Searches the graph for existing data related to entities from a batch,
        respecting actor contexts by searching per actor.

        Args:
            entity_info_list (list[dict]): List of extracted entity information,
                                         each like {"name": ..., "type": ..., "actor_id": ...}.
            base_filters (dict): Base filters for the search, typically containing
                                 the main user_id or session_id.

        Returns:
            list[dict]: An aggregated list of search results (relationships)
                        from all per-actor searches.
        """
        if not entity_info_list:
            logger.info("No entities provided for batched graph search.")
            return []

        comprehensive_search_output = []

        # Group entities by actor_id
        entities_by_actor = {}
        for entity_info in entity_info_list:
            actor_id = entity_info.get("actor_id", "unknown_actor_batch") # Fallback if actor_id is missing
            if actor_id not in entities_by_actor:
                entities_by_actor[actor_id] = []
            # We need a list of entity names for _search_graph_db's node_list argument
            entities_by_actor[actor_id].append(entity_info["name"])

        for actor_id, actor_entity_names in entities_by_actor.items():
            if not actor_entity_names:
                continue

            # Create specific filters for this actor's search
            actor_specific_filters = base_filters.copy()
            actor_specific_filters["actor_id"] = actor_id
            # Ensure user_id (or primary session id) is present
            if "user_id" not in actor_specific_filters and "run_id" not in actor_specific_filters and "agent_id" not in actor_specific_filters:
                # This case should ideally be covered by base_filters having the session key
                logger.warning(f"Missing primary session ID in base_filters for actor {actor_id}")
                # Fallback to a default if absolutely necessary, though risky
                # actor_specific_filters["user_id"] = base_filters.get("user_id", "default_user_search")
                # It's better to ensure base_filters is correctly populated upstream.

            # Remove duplicates from entity names for the current actor before searching
            unique_actor_entity_names = sorted(list(set(actor_entity_names)))

            logger.debug(f"Searching graph for actor '{actor_id}' with entities: {unique_actor_entity_names}")
            try:
                # Call the existing single-actor search method
                search_output_for_actor = self._search_graph_db(
                    node_list=unique_actor_entity_names,
                    filters=actor_specific_filters
                )
                if search_output_for_actor:
                    comprehensive_search_output.extend(search_output_for_actor)
            except Exception as e:
                logger.error(f"Error during graph search for actor {actor_id}: {e}")

        # Deduplicate results if needed, as different initial nodes might lead to the same relationships
        # The current _search_graph_db already uses DISTINCT in Cypher, but across multiple calls,
        # there could still be overlaps if we are not careful with what makes a relationship unique.
        # A simple deduplication based on all key fields of a relationship dict:
        deduplicated_results = []
        seen_relations = set()
        for rel in comprehensive_search_output:
            # Create a frozenset of items to make the dict hashable for set storage
            rel_tuple = tuple(sorted(rel.items()))
            if rel_tuple not in seen_relations:
                deduplicated_results.append(rel)
                seen_relations.add(rel_tuple)

        logger.debug(f"Batched graph search found {len(deduplicated_results)} unique relationships.")
        return deduplicated_results

    def _get_delete_entities_from_search_output_batch(self, comprehensive_search_output, consolidated_text, actor_segments, base_user_id_for_prompt):
        """
        Gets entities to be deleted based on a comprehensive search output and consolidated new text,
        using a single LLM call.

        Args:
            comprehensive_search_output (list[dict]): Aggregated list of existing relevant relationships.
            consolidated_text (str): The single string of all new messages, with actor prefixes.
            actor_segments (list[dict]): Information about actor segments, to help LLM understand context.
                                       (Currently unused in prompt, but available for future refinement)
            base_user_id_for_prompt (str): A general user/session ID to use if a global self-reference ID is needed in the prompt.
                                           The prompt primarily relies on actor prefixes in consolidated_text.

        Returns:
            list[dict]: A list of relationship dictionaries to be deleted.
        """
        if not comprehensive_search_output:
            logger.info("No existing memories to evaluate for deletion.")
            return []

        search_output_string = format_entities(comprehensive_search_output) # format_entities is existing helper

        # The system prompt for deletion needs to be robust for multi-actor context.
        # The self_reference_id here is a general fallback; the LLM is guided more by actor prefixes in the text.
        # We use base_user_id_for_prompt as a generic placeholder if the prompt template needs one,
        # but the core instruction is to use actor prefixes from the `consolidated_text`.

        system_prompt_template = DELETE_RELATIONS_SYSTEM_PROMPT # Existing prompt template

        # The existing DELETE_RELATIONS_SYSTEM_PROMPT uses "SELF_REFERENCE_ID".
        # For batched context, direct self-reference replacement is tricky.
        # The prompt now more heavily relies on the LLM understanding actor prefixes in the consolidated_text.
        # We provide a general ID here, but the instructions within the prompt emphasize actor prefixes.
        # This might need further refinement based on LLM behavior.
        system_prompt = system_prompt_template.replace("SELF_REFERENCE_ID", base_user_id_for_prompt)
        system_prompt += ("\n\nIMPORTANT: The 'New Information' below contains messages from multiple actors, prefixed by their names (e.g., 'actor_name: message'). "
                          "Evaluate deletions considering who said what. Self-references like 'I' or 'my' pertain to the prefixed actor of that specific message segment.")

        user_prompt = f"Here are the existing relevant graph memories:\n{search_output_string}\n\nNew Information (potentially from multiple actors):\n{consolidated_text}"

        _tools = [DELETE_MEMORY_TOOL_GRAPH]
        if self.llm_provider in ["azure_openai_structured", "openai_structured"]:
            _tools = [DELETE_MEMORY_STRUCT_TOOL_GRAPH]

        memory_updates_response = self.llm.generate_response(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            tools=_tools,
        )

        to_be_deleted_batch = []
        try:
            if memory_updates_response.get("tool_calls"):
                for tool_call in memory_updates_response["tool_calls"]:
                    if tool_call["name"] == "delete_graph_memory":
                        args = tool_call["arguments"]
                        if isinstance(args, str):
                            try:
                                args = json.loads(args)
                            except json.JSONDecodeError:
                                logger.error(f"Failed to parse string arguments for delete_graph_memory: {args}")
                                continue
                        # The existing _remove_spaces_from_entities expects a list of such dicts.
                        # Here, args is a single dict, so we wrap it in a list.
                        # This helper also lowercases source, relationship, destination.
                        cleaned_args_list = self._remove_spaces_from_entities(deepcopy([args]))
                        if cleaned_args_list:
                            to_be_deleted_batch.append(cleaned_args_list[0])
        except Exception as e:
            logger.exception(
                f"Error in parsing batched deletion results: {e}, response: {memory_updates_response}"
            )

        logger.debug(f"Batched to_be_deleted: {to_be_deleted_batch}")
        return to_be_deleted_batch
