"""
Parte 3 - Agente conversacional con LangGraph.

Integra:
- TicketClassifier (Parte 1.1) -> clasificar la intención del mensaje
- ChurnPredictor (Parte 1.2)   -> evaluar riesgo del cliente
- SentimentClassifier (Parte 2.1) -> detectar frustración -> escalar

Diseño: el agente NO depende de un LLM externo (no requiere API key) --
usa lógica determinística + los modelos entrenados. Al final del archivo
se explica cómo conectar un LLM real si se desea.
"""
from typing import TypedDict, Optional, List, Dict, Any
from langgraph.graph import StateGraph, END

from src.ml.ticket_classifier import TicketClassifier
from src.ml.churn_predictor import ChurnPredictor
from src.dl.sentiment_model import SentimentClassifier


# ---------------------------------------------------------------------------
# Estado del agente (requisito del enunciado)
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    messages: List[Dict[str, str]]      # historial [{"role": "user"/"assistant", "content": ...}]
    customer_id: Optional[str]
    intent: Optional[str]
    context: Dict[str, Any]             # info del cliente si se identifica
    escalate: bool
    response: str


# ---------------------------------------------------------------------------
# "Base de datos" simulada de clientes (en la API real, esto sería SQLAlchemy)
# ---------------------------------------------------------------------------

SIMULATED_CUSTOMERS = {
    "1001": {
        "name": "María Fernanda Rojas", "plan_type": "premium", "monthly_charge": 55.0,
        "tenure_months": 60, "total_charges": 3300.0, "contract_type": "two_year",
        "payment_method": "credit_card", "num_tickets": 0, "avg_satisfaction": 4.7,
    },
    "1002": {
        "name": "Jorge Andrés Peña", "plan_type": "basic", "monthly_charge": 18.0,
        "tenure_months": 2, "total_charges": 36.0, "contract_type": "month-to-month",
        "payment_method": "cash", "num_tickets": 4, "avg_satisfaction": 1.8,
    },
}

_ticket_classifier = None
_churn_predictor = None
_sentiment_classifier = None


def _get_models():
    """Carga perezosa (lazy) de los modelos -- solo se cargan la primera vez
    que se usan, no al importar el módulo (evita cargar 3 modelos pesados
    de TensorFlow/sklearn solo por importar el agente)."""
    global _ticket_classifier, _churn_predictor, _sentiment_classifier
    if _ticket_classifier is None:
        _ticket_classifier = TicketClassifier()
        _churn_predictor = ChurnPredictor()
        _sentiment_classifier = SentimentClassifier()
    return _ticket_classifier, _churn_predictor, _sentiment_classifier


GREETING_WORDS = {"hola", "buenos dias", "buenas tardes", "buenas noches", "buen dia", "buenas"}
FAREWELL_WORDS = {"adios", "gracias hasta luego", "hasta luego", "chao", "nos vemos", "eso es todo gracias"}


def _normalize(text: str) -> str:
    import unicodedata
    text = text.lower().strip()
    text = "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")
    return text


# ---------------------------------------------------------------------------
# NODOS
# ---------------------------------------------------------------------------

def classify_intent(state: AgentState) -> AgentState:
    last_message = state["messages"][-1]["content"]
    normalized = _normalize(last_message)

    if any(word in normalized for word in GREETING_WORDS) and len(normalized) < 25:
        state["intent"] = "greeting"
        return state
    if any(word in normalized for word in FAREWELL_WORDS):
        state["intent"] = "farewell"
        return state

    ticket_classifier, _, _ = _get_models()
    try:
        resultado = ticket_classifier.predict(last_message)
        categoria = resultado["category"]
    except ValueError:
        # Manejo de estado inválido: texto muy corto para el clasificador
        state["intent"] = "invalid"
        return state

    intent_map = {
        "TECH": "technical_support",
        "BILL": "account_query",
        "PLAN": "account_query",
        "CNCL": "account_query",
        "OTHR": "general_info",
    }
    state["intent"] = intent_map.get(categoria, "general_info")
    state["context"]["detected_category"] = categoria
    return state


def get_customer_info(state: AgentState) -> AgentState:
    customer_id = state.get("customer_id")
    if customer_id and customer_id in SIMULATED_CUSTOMERS:
        customer = SIMULATED_CUSTOMERS[customer_id]
        state["context"]["customer"] = customer

        _, churn_predictor, _ = _get_models()
        churn_result = churn_predictor.predict(customer)
        state["context"]["churn_risk"] = churn_result["risk_level"]
        state["context"]["churn_probability"] = churn_result["churn_probability"]
    return state


def handle_account_query(state: AgentState) -> AgentState:
    customer = state["context"].get("customer")
    categoria = state["context"].get("detected_category", "")

    if categoria == "CNCL":
        if customer and state["context"].get("churn_risk") == "alto":
            texto = (
                f"Entiendo que deseas cancelar. Antes de proceder, quiero mencionarte que "
                f"como cliente {customer['plan_type']} podríamos ofrecerte alternativas. "
                f"Voy a conectarte con un asesor de retención."
            )
            state["escalate"] = True
        else:
            texto = "Entiendo que deseas cancelar el servicio. Te comparto el proceso y los tiempos estimados."
    elif customer:
        texto = (
            f"Hola {customer['name'].split()[0]}, veo que tienes el plan {customer['plan_type']} "
            f"con un cargo mensual de ${customer['monthly_charge']:.2f}. ¿En qué más te puedo ayudar con tu cuenta?"
        )
    else:
        texto = "Con gusto te ayudo con tu consulta de cuenta. ¿Podrías indicarme tu número de cliente?"

    state["response"] = texto
    return state


def handle_technical_support(state: AgentState) -> AgentState:
    state["response"] = (
        "Lamento el inconveniente técnico. He registrado tu caso y nuestro equipo "
        "está revisando la conexión. Mientras tanto, ¿podrías confirmarme si el problema "
        "es constante o intermitente?"
    )
    return state


def handle_general_info(state: AgentState) -> AgentState:
    state["response"] = (
        "Con gusto te comparto la información. Nuestros canales de atención están "
        "disponibles de lunes a sábado. ¿Hay algo más específico que quieras consultar?"
    )
    return state


def check_escalation(state: AgentState) -> AgentState:
    last_message = state["messages"][-1]["content"]
    _, _, sentiment_classifier = _get_models()

    try:
        sentiment_result = sentiment_classifier.predict(last_message)
        state["context"]["sentiment"] = sentiment_result["sentiment"]
        if sentiment_result["sentiment"] == "negativo" and sentiment_result["confidence"] > 0.6:
            state["escalate"] = True
    except Exception:
        # Manejo de errores: si el modelo de sentimiento falla, no rompemos el flujo
        state["context"]["sentiment"] = "desconocido"

    return state


def generate_response(state: AgentState) -> AgentState:
    if state["intent"] == "greeting":
        state["response"] = "¡Hola! Soy el asistente virtual. ¿En qué puedo ayudarte hoy?"
        return state

    if state["intent"] == "farewell":
        state["response"] = "¡Gracias por contactarnos! Que tengas un excelente día."
        return state

    if state["intent"] == "invalid":
        state["response"] = (
            "Disculpa, ¿podrías darme un poco más de detalle sobre tu consulta? "
            "Necesito al menos una breve descripción para poder ayudarte."
        )
        return state

    if state.get("escalate"):
        state["response"] = (
            state["response"]
            + " He notado que este caso requiere atención especial, así que te voy "
              "a transferir con un agente humano para resolverlo mejor."
        )

    return state


# ---------------------------------------------------------------------------
# Enrutamiento condicional
# ---------------------------------------------------------------------------

def route_after_classify(state: AgentState) -> str:
    if state["intent"] in ("greeting", "farewell", "invalid"):
        return "generate_response"
    return "get_customer_info"


def route_after_customer_info(state: AgentState) -> str:
    return {
        "technical_support": "handle_technical_support",
        "account_query": "handle_account_query",
        "general_info": "handle_general_info",
    }[state["intent"]]


# ---------------------------------------------------------------------------
# Construcción del grafo
# ---------------------------------------------------------------------------

def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("classify_intent", classify_intent)
    graph.add_node("get_customer_info", get_customer_info)
    graph.add_node("handle_account_query", handle_account_query)
    graph.add_node("handle_technical_support", handle_technical_support)
    graph.add_node("handle_general_info", handle_general_info)
    graph.add_node("check_escalation", check_escalation)
    graph.add_node("generate_response", generate_response)

    graph.set_entry_point("classify_intent")

    graph.add_conditional_edges(
        "classify_intent",
        route_after_classify,
        {"get_customer_info": "get_customer_info", "generate_response": "generate_response"},
    )
    graph.add_conditional_edges(
        "get_customer_info",
        route_after_customer_info,
        {
            "handle_technical_support": "handle_technical_support",
            "handle_account_query": "handle_account_query",
            "handle_general_info": "handle_general_info",
        },
    )

    graph.add_edge("handle_account_query", "check_escalation")
    graph.add_edge("handle_technical_support", "check_escalation")
    graph.add_edge("handle_general_info", "check_escalation")
    graph.add_edge("check_escalation", "generate_response")
    graph.add_edge("generate_response", END)

    return graph.compile()


class CustomerServiceAgent:
    """Wrapper de alto nivel -- lo que importará la API FastAPI."""

    def __init__(self):
        self.graph = build_graph()
        self.sessions: Dict[str, AgentState] = {}

    def chat(self, session_id: str, message: str, customer_id: Optional[str] = None) -> dict:
        state = self.sessions.get(session_id, {
            "messages": [], "customer_id": customer_id, "intent": None,
            "context": {}, "escalate": False, "response": "",
        })
        state["messages"].append({"role": "user", "content": message})
        state["escalate"] = False  # se reevalúa en cada turno, no debe heredarse del anterior
        if customer_id:
            state["customer_id"] = customer_id

        result_state = self.graph.invoke(state)
        result_state["messages"].append({"role": "assistant", "content": result_state["response"]})
        self.sessions[session_id] = result_state

        return {
            "response": result_state["response"],
            "intent": result_state["intent"],
            "escalate": result_state["escalate"],
            "context": result_state["context"],
        }


_agent_instance: Optional["CustomerServiceAgent"] = None


def get_agent_instance() -> "CustomerServiceAgent":
    """Singleton compartido -- tanto la API REST (routers/agent.py) como el
    servidor MCP (src/mcp/server.py) deben usar ESTA misma instancia, para
    que una conversación iniciada por un canal pueda continuar por el otro
    (las sesiones viven en memoria dentro del objeto agent)."""
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = CustomerServiceAgent()
    return _agent_instance


if __name__ == "__main__":
    agent = CustomerServiceAgent()

    print("=" * 60)
    print("DEMO DEL AGENTE CONVERSACIONAL")
    print("=" * 60)

    conversaciones_demo = [
        ("sesion_1", "Hola", None),
        ("sesion_1", "Mi internet lleva 3 días sin funcionar y estoy muy molesto", "1002"),
        ("sesion_1", "Gracias, hasta luego", None),
        ("sesion_2", "Quiero cancelar mi servicio", "1002"),
        ("sesion_3", "no", None),  # texto inválido -- prueba de manejo de errores
    ]

    for session_id, mensaje, customer_id in conversaciones_demo:
        resultado = agent.chat(session_id, mensaje, customer_id)
        print(f"\n[{session_id}] Usuario: {mensaje}")
        print(f"  Intent: {resultado['intent']} | Escalar: {resultado['escalate']}")
        print(f"  Agente: {resultado['response']}")
