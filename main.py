from agent import ImageState, orchestra, story_writer, image_generator, routing
from langgraph.graph import StateGraph, START, END
import dotenv
dotenv.load_dotenv()

workflow = StateGraph(ImageState)
workflow.add_node("orchestra", orchestra)
workflow.add_node("story_writer", story_writer)
workflow.add_node("image_generator", image_generator)
workflow.add_edge(START, "orchestra")
workflow.add_edge("orchestra", "story_writer")
workflow.add_conditional_edges("story_writer", routing, {
    "generate_image": "image_generator",
    "skip_image": END
})
workflow.add_edge("image_generator", END)
workflow = workflow.compile()