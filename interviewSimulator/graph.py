from langgraph.graph import StateGraph, START, END
from .agents import interviewer, evaluator, coach
from IPython.display import Image, display

# ─────────────────────────────────────────────
# Nodes
# ─────────────────────────────────────────────

def interviewer_node(state):
    return interviewer(state)


def evaluator_node(state):
    return evaluator(state)


def coach_node(state):
    return coach(state)


# ─────────────────────────────────────────────
# Router (DECIDES FLOW)
# ─────────────────────────────────────────────

def router(state):
    # If interview finished → END
    if not state.get("active", True):
        return END

    # Otherwise → ask next question
    return "interviewer"

def starting_node(state):
    if state.get("user_answer", ""):
        return "evaluator"
    
    return "interviewer"

# ─────────────────────────────────────────────
# Build Graph
# ─────────────────────────────────────────────

def build_graph():
    graph = StateGraph(dict)

    graph.add_node("interviewer", interviewer_node)
    graph.add_node("evaluator", evaluator_node)
    graph.add_node("coach", coach_node)

    # ENTRY
    # graph.set_entry_point(START)

    # FLOW
    graph.add_conditional_edges(START, starting_node)
    graph.add_edge("interviewer", END)  # after question → wait for user

    graph.add_edge("evaluator", "coach")
    graph.add_conditional_edges("coach", router)

    graphRunnable = graph.compile()
    # show_graph()
    return graphRunnable

graph = build_graph()


def show_graph():
    img = graph.get_graph().draw_mermaid_png()

    with open("graph.png", "wb") as f:
        f.write(img)

    print("Graph saved as graph.png")

def run_interview_graph(state, action):
    if action == "start":
        # Only generate first question
        return graph.invoke(state)

    elif action == "answer":
        # Continue flow from evaluator
        return graph.invoke(state, config={"start_at": "evaluator"})

    return state