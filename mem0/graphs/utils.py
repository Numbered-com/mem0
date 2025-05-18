UPDATE_GRAPH_PROMPT = """
You are an AI expert specializing in graph memory management and optimization. Your task is to analyze existing graph memories alongside new information, and update the relationships in the memory list to ensure the most accurate, current, and coherent representation of knowledge.

Input:
1. Existing Graph Memories: A list of current graph memories, each containing source, target, and relationship information.
2. New Graph Memory: Fresh information to be integrated into the existing graph structure.

Guidelines:
1. Identification: Use the source and target as primary identifiers when matching existing memories with new information.
2. Conflict Resolution:
   - If new information contradicts an existing memory:
     a) For matching source and target but differing content, update the relationship of the existing memory.
     b) If the new memory provides more recent or accurate information, update the existing memory accordingly.
3. Comprehensive Review: Thoroughly examine each existing graph memory against the new information, updating relationships as necessary. Multiple updates may be required.
4. Consistency: Maintain a uniform and clear style across all memories. Each entry should be concise yet comprehensive.
5. Semantic Coherence: Ensure that updates maintain or improve the overall semantic structure of the graph.
6. Temporal Awareness: If timestamps are available, consider the recency of information when making updates.
7. Relationship Refinement: Look for opportunities to refine relationship descriptions for greater precision or clarity.
8. Redundancy Elimination: Identify and merge any redundant or highly similar relationships that may result from the update.

Memory Format:
source -- RELATIONSHIP -- destination

Task Details:
======= Existing Graph Memories:=======
{existing_memories}

======= New Graph Memory:=======
{new_memories}

Output:
Provide a list of update instructions, each specifying the source, target, and the new relationship to be set. Only include memories that require updates.
"""

EXTRACT_RELATIONS_PROMPT = """

You are an advanced algorithm designed to extract structured information from text to construct knowledge graphs. Your goal is to capture comprehensive and accurate information. Follow these key principles:

1. Extract only explicitly stated information from the text.
2. Establish relationships among the entities provided.
3. Use "USER_ID" as the source entity for any self-references (e.g., "I," "me," "my," etc.) in user messages.
CUSTOM_PROMPT

Relationships:
    - Use consistent, general, and timeless relationship types.
    - Example: Prefer "professor" over "became_professor."
    - Relationships should only be established among the entities explicitly mentioned in the user message.

Entity Consistency:
    - Ensure that relationships are coherent and logically align with the context of the message.
    - Maintain consistent naming for entities across the extracted data.

Strive to construct a coherent and easily understandable knowledge graph by eshtablishing all the relationships among the entities and adherence to the user's context.

Adhere strictly to these guidelines to ensure high-quality knowledge graph extraction."""

DELETE_RELATIONS_SYSTEM_PROMPT = """
You are a graph memory manager specializing in identifying, managing, and optimizing relationships within graph-based memories. Your primary task is to analyze a list of existing relationships and determine which ones should be deleted based on the new information provided.

Input:
1. Existing Graph Memories: A list of current graph memories, each containing source, relationship, and destination information.
2. New Text: The new information to be integrated into the existing graph structure.
3. Use "SELF_REFERENCE_ID" as node for any self-references (e.g., "I," "me," "my," etc.) in user messages.

Guidelines:
1. Identify Redundancy: Delete relationships that are exact duplicates of newly stated information if the new information implies a reset or a more current state.
2. Identify Contradictions: If the new text directly contradicts an existing relationship, the existing relationship should be deleted. For example:
    - If existing is "person_A -- status -- single" and new info is "person_A is married", delete "person_A -- status -- single".
    - If existing is "person_A -- is_friend_of -- person_B" and new info is "person_A is_not_friend_of person_B", delete "person_A -- is_friend_of -- person_B".
    - If existing is "event_X -- status -- upcoming" and new info is "event_X happened last week", delete "event_X -- status -- upcoming".
3. Identify Obsolescence: If new information provides an update that makes an old piece of information obsolete, delete the old one. For example:
    - If existing is "employee_Y -- works_at -- company_Z" and new info is "employee_Y now works_at company_W", delete "employee_Y -- works_at -- company_Z".
    - If existing is "project_P -- due_date -- date_D1" and new info is "project_P due_date is now date_D2", delete "project_P -- due_date -- date_D1".
4. Focus on Explicit Information: Only delete relationships if the new text strongly implies a deletion due to contradiction or clear superseding. Do not infer deletions too liberally.
5. Relationship Specificity: Consider if a more specific relationship in the new text replaces a more general one.

Memory Format:
source -- relationship -- destination

Provide a list of deletion instructions.
For each relationship you identify as needing deletion according to the guidelines, you MUST make one call to the `delete_graph_memory` tool.
If multiple relationships need to be deleted, you MUST make multiple, separate calls to the `delete_graph_memory` tool (one call per relationship).
Provide the `source`, `relationship`, and `destination` for each deletion using this tool.
"""


def get_delete_messages(existing_memories_string, data, self_reference_id):
    return DELETE_RELATIONS_SYSTEM_PROMPT.replace(
        "SELF_REFERENCE_ID", self_reference_id
    ), f"Here are the existing memories: {existing_memories_string} \n\n New Information: {data}"
