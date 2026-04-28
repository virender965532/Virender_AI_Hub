from langgraph.graph import StateGraph, END
from .agents import interviewer, evaluator, coach


def interviewer_node(state):
    return interviewer(state)


def evaluator_node(state):
    return evaluator(state)


def coach_node(state):
    return coach(state)


def router(state):
    if not state["active"]:
        return END
    return "interviewer"


def build_graph():
    graph = StateGraph(dict)

    graph.add_node("interviewer", interviewer_node)
    graph.add_node("evaluator", evaluator_node)
    graph.add_node("coach", coach_node)

    graph.set_entry_point("interviewer")

    graph.add_edge("interviewer", "evaluator")
    graph.add_edge("evaluator", "coach")
    graph.add_edge("coach", END)

    return graph.compile()


graph = build_graph()


def run_interview_graph(state, action):
    if action == "start":
        return interviewer(state)

    if action == "answer":
        state = evaluator(state)
        state = coach(state)
        state = interviewer(state)
        return state

    return state