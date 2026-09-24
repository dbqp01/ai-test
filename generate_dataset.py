from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any


SYSTEM_PROMPT = """You are the decision policy inside a universal website operator.
Choose the next step from the observed actions for the user's goal.
Return exactly one JSON object and no markdown.
Allowed decisions: act, request_confirmation, finish, block.
Use action_index only when decision is act or request_confirmation.
Never claim finish without observable evidence. Never perform a sensitive write without write_confirmed=true.
If a required file, credential, or prerequisite is missing, choose block and explain the missing prerequisite.
"""


LABELS = {
    "es": {
        "home": ["Inicio", "Panel principal", "Página de inicio"],
        "courses": ["Cursos", "Mis cursos", "Clases"],
        "assignments": ["Tareas pendientes", "Tareas", "Actividades por entregar"],
        "algebra": ["Álgebra", "Curso de Álgebra", "Álgebra básica"],
        "assignment": ["Tarea: ecuaciones lineales", "Álgebra — ecuaciones lineales", "Actividad de ecuaciones"],
        "open": ["Abrir", "Ver", "Continuar"],
        "back": ["Volver", "Atrás", "Regresar"],
        "upload": ["Archivo de solución", "Subir solución", "Seleccionar PDF"],
        "submit": ["Entregar tarea", "Enviar actividad", "Presentar solución"],
        "reports": ["Reportes", "Informes", "Analítica"],
        "report": ["Reporte mensual", "Informe de progreso", "Resumen de actividad"],
        "export": ["Exportar CSV", "Descargar reporte", "Generar archivo"],
        "support": ["Soporte", "Ayuda", "Centro de ayuda"],
        "new_ticket": ["Nuevo ticket", "Crear solicitud", "Contactar soporte"],
        "subject": ["Asunto", "Título de la solicitud", "Tema"],
        "message": ["Mensaje", "Descripción", "Detalles"],
        "send": ["Enviar solicitud", "Crear ticket", "Enviar mensaje"],
        "products": ["Catálogo", "Productos", "Tienda"],
        "product": ["Curso de Python", "Libro de programación", "Plan profesional"],
        "cart": ["Carrito", "Mi compra", "Cesta"],
        "checkout": ["Finalizar compra", "Pagar", "Continuar al pago"],
        "start": ["Iniciar servicio", "Encender instancia", "Activar"],
        "instances": ["Instancias", "Servidores", "Recursos"],
        "instance": ["Servidor de desarrollo", "Instancia GPU", "Entorno de trabajo"],
        "goal_edu": ["Entrega la tarea de Álgebra usando este archivo", "Sube y entrega la actividad de ecuaciones", "Presenta mi solución de Álgebra"],
        "goal_report": ["Descarga el reporte mensual", "Exporta el informe de progreso", "Obtén el CSV de analítica"],
        "goal_ticket": ["Crea un ticket de soporte con este mensaje", "Envía una solicitud a soporte", "Abre un caso con la descripción indicada"],
        "goal_shop": ["Compra el producto seleccionado", "Completa la compra del curso", "Finaliza el pedido"],
        "goal_cloud": ["Inicia mi instancia de desarrollo", "Enciende el servidor GPU", "Activa el entorno de trabajo"],
        "missing": ["Falta un requisito", "No se puede continuar", "Se necesita información"],
        "expired": ["La sesión expiró", "Tu sesión ha caducado", "Vuelve a iniciar sesión"],
        "success": ["Operación completada", "Éxito", "Confirmación"],
    },
    "en": {
        "home": ["Home", "Main dashboard", "Start page"],
        "courses": ["Courses", "My courses", "Classes"],
        "assignments": ["Pending assignments", "Assignments", "Due work"],
        "algebra": ["Algebra", "Algebra course", "Basic algebra"],
        "assignment": ["Linear equations assignment", "Algebra equations", "Equations activity"],
        "open": ["Open", "View", "Continue"],
        "back": ["Back", "Return", "Go back"],
        "upload": ["Solution file", "Upload solution", "Select PDF"],
        "submit": ["Submit assignment", "Send work", "Submit solution"],
        "reports": ["Reports", "Analytics", "Insights"],
        "report": ["Monthly report", "Progress report", "Activity summary"],
        "export": ["Export CSV", "Download report", "Generate file"],
        "support": ["Support", "Help", "Help center"],
        "new_ticket": ["New ticket", "Create request", "Contact support"],
        "subject": ["Subject", "Request title", "Topic"],
        "message": ["Message", "Description", "Details"],
        "send": ["Send request", "Create ticket", "Send message"],
        "products": ["Catalog", "Products", "Store"],
        "product": ["Python course", "Programming book", "Professional plan"],
        "cart": ["Cart", "My purchase", "Basket"],
        "checkout": ["Checkout", "Pay", "Continue to payment"],
        "start": ["Start service", "Start instance", "Activate"],
        "instances": ["Instances", "Servers", "Resources"],
        "instance": ["Development server", "GPU instance", "Work environment"],
        "goal_edu": ["Submit the Algebra assignment using this file", "Upload and submit the equations activity", "Submit my Algebra solution"],
        "goal_report": ["Download the monthly report", "Export the progress report", "Get the analytics CSV"],
        "goal_ticket": ["Create a support ticket with this message", "Send a request to support", "Open a case with the provided description"],
        "goal_shop": ["Buy the selected product", "Complete the course purchase", "Finish the order"],
        "goal_cloud": ["Start my development instance", "Turn on the GPU server", "Activate the work environment"],
        "missing": ["A prerequisite is missing", "Cannot continue", "More information is required"],
        "expired": ["The session expired", "Your session has expired", "Sign in again"],
        "success": ["Operation completed", "Success", "Confirmation"],
    },
}


def pick(rng: random.Random, lang: str, key: str) -> str:
    return rng.choice(LABELS[lang][key])


def action(
    ident: str,
    kind: str,
    label: str,
    next_state: str | None,
    *,
    dangerous: bool = False,
    input_type: str = "",
    name: str = "",
    placeholder: str = "",
    enabled: bool = True,
) -> dict[str, Any]:
    return {
        "id": ident,
        "kind": kind,
        "label": label,
        "role": "button" if kind == "click" else "textbox",
        "input_type": input_type,
        "name": name,
        "placeholder": placeholder,
        "dangerous": dangerous,
        "enabled": enabled,
        "next": next_state,
    }


def site_template(family: str, lang: str, rng: random.Random) -> tuple[dict[str, dict[str, Any]], list[list[str]], str, dict[str, Any]]:
    """Return states, valid routes, a goal and its default context."""
    if family == "education":
        states = {
            "home": {"url": "/home", "title": pick(rng, lang, "home"), "text": "Portal educativo. Elige una sección.", "actions": [
                action("courses", "click", pick(rng, lang, "courses"), "courses"),
                action("assignments", "click", pick(rng, lang, "assignments"), "assignments"),
                action("profile", "click", "Perfil" if lang == "es" else "Profile", "home"),
            ]},
            "courses": {"url": "/courses", "title": pick(rng, lang, "courses"), "text": "Lista de cursos disponibles.", "actions": [
                action("algebra", "click", pick(rng, lang, "algebra"), "algebra"),
                action("home", "click", pick(rng, lang, "home"), "home"),
            ]},
            "algebra": {"url": "/courses/algebra", "title": pick(rng, lang, "algebra"), "text": "Curso de Álgebra básica. Hay una actividad pendiente.", "actions": [
                action("assignment", "click", pick(rng, lang, "assignment"), "assignment_empty"),
                action("courses", "click", pick(rng, lang, "courses"), "courses"),
            ]},
            "assignments": {"url": "/assignments", "title": pick(rng, lang, "assignments"), "text": "Actividades pendientes del estudiante.", "actions": [
                action("assignment", "click", pick(rng, lang, "assignment"), "assignment_empty"),
                action("courses", "click", pick(rng, lang, "courses"), "courses"),
            ]},
            "assignment_empty": {"url": "/assignments/algebra", "title": pick(rng, lang, "assignment"), "text": "Sube tu solución en PDF y entrégala.", "actions": [
                action("upload", "input", pick(rng, lang, "upload"), "assignment_ready", input_type="file", name="solution", placeholder="PDF"),
                action("submit", "click", pick(rng, lang, "submit"), "submitted", dangerous=True),
                action("back", "click", pick(rng, lang, "back"), "assignments"),
            ]},
            "assignment_ready": {"url": "/assignments/algebra", "title": pick(rng, lang, "assignment"), "text": "Archivo seleccionado. La entrega tendrá efectos externos.", "actions": [
                action("submit", "click", pick(rng, lang, "submit"), "submitted", dangerous=True),
                action("back", "click", pick(rng, lang, "back"), "assignments"),
            ]},
            "submitted": {"url": "/assignments/algebra/submitted", "title": pick(rng, lang, "success"), "text": "Tarea entregada. Confirmation ID: EDU-83921", "actions": []},
        }
        return states, [["assignments", "assignment", "upload", "submit"], ["courses", "algebra", "assignment", "upload", "submit"]], pick(rng, lang, "goal_edu"), {"file": "solution.pdf"}

    if family == "report":
        states = {
            "home": {"url": "/dashboard", "title": pick(rng, lang, "home"), "text": "Panel de trabajo con accesos a informes.", "actions": [
                action("reports", "click", pick(rng, lang, "reports"), "reports"),
                action("profile", "click", "Perfil" if lang == "es" else "Profile", "home"),
            ]},
            "reports": {"url": "/reports", "title": pick(rng, lang, "reports"), "text": "Informes generados y disponibles.", "actions": [
                action("report", "click", pick(rng, lang, "report"), "report"),
                action("home", "click", pick(rng, lang, "home"), "home"),
            ]},
            "report": {"url": "/reports/monthly", "title": pick(rng, lang, "report"), "text": "Reporte mensual listo para exportar.", "actions": [
                action("export", "click", pick(rng, lang, "export"), "exported", dangerous=True),
                action("back", "click", pick(rng, lang, "back"), "reports"),
            ]},
            "exported": {"url": "/reports/monthly/download", "title": pick(rng, lang, "success"), "text": "Reporte exportado. Download ID: CSV-4812", "actions": []},
        }
        return states, [["reports", "report", "export"]], pick(rng, lang, "goal_report"), {}

    if family == "support":
        states = {
            "home": {"url": "/dashboard", "title": pick(rng, lang, "home"), "text": "Panel con acceso a soporte.", "actions": [
                action("support", "click", pick(rng, lang, "support"), "support"),
                action("profile", "click", "Perfil" if lang == "es" else "Profile", "home"),
            ]},
            "support": {"url": "/support", "title": pick(rng, lang, "support"), "text": "Consulta artículos o crea una solicitud.", "actions": [
                action("new_ticket", "click", pick(rng, lang, "new_ticket"), "ticket_empty"),
                action("home", "click", pick(rng, lang, "home"), "home"),
            ]},
            "ticket_empty": {"url": "/support/new", "title": pick(rng, lang, "new_ticket"), "text": "Completa el asunto y la descripción.", "actions": [
                action("subject", "input", pick(rng, lang, "subject"), "ticket_subject", input_type="text", name="subject"),
                action("message", "input", pick(rng, lang, "message"), "ticket_ready", input_type="text", name="message"),
                action("send", "click", pick(rng, lang, "send"), "ticket_submitted", dangerous=True, enabled=False),
            ]},
            "ticket_subject": {"url": "/support/new", "title": pick(rng, lang, "new_ticket"), "text": "Asunto completado; falta la descripción.", "actions": [
                action("message", "input", pick(rng, lang, "message"), "ticket_ready", input_type="text", name="message"),
                action("subject", "input", pick(rng, lang, "subject"), "ticket_subject", input_type="text", name="subject"),
                action("send", "click", pick(rng, lang, "send"), "ticket_submitted", dangerous=True, enabled=False),
            ]},
            "ticket_ready": {"url": "/support/new", "title": pick(rng, lang, "new_ticket"), "text": "Solicitud completa. Enviar creará un ticket externo.", "actions": [
                action("send", "click", pick(rng, lang, "send"), "ticket_submitted", dangerous=True),
                action("message", "input", pick(rng, lang, "message"), "ticket_ready", input_type="text", name="message"),
            ]},
            "ticket_submitted": {"url": "/support/ticket/1001", "title": pick(rng, lang, "success"), "text": "Solicitud creada. Ticket ID: SUP-1001", "actions": []},
        }
        return states, [["support", "new_ticket", "subject", "message", "send"]], pick(rng, lang, "goal_ticket"), {"inputs": {"subject": "No puedo abrir el curso", "message": "El enlace de la actividad devuelve un error."}}

    if family == "shop":
        states = {
            "home": {"url": "/", "title": pick(rng, lang, "home"), "text": "Catálogo de productos.", "actions": [
                action("products", "click", pick(rng, lang, "products"), "products"),
                action("cart", "click", pick(rng, lang, "cart"), "cart"),
            ]},
            "products": {"url": "/products", "title": pick(rng, lang, "products"), "text": "Productos disponibles.", "actions": [
                action("product", "click", pick(rng, lang, "product"), "product"),
                action("home", "click", pick(rng, lang, "home"), "home"),
            ]},
            "product": {"url": "/products/1", "title": pick(rng, lang, "product"), "text": "Producto seleccionado.", "actions": [
                action("add_to_cart", "click", "Añadir al carrito" if lang == "es" else "Add to cart", "cart_ready"),
                action("products", "click", pick(rng, lang, "products"), "products"),
            ]},
            "cart_ready": {"url": "/cart", "title": pick(rng, lang, "cart"), "text": "Tu carrito contiene un producto.", "actions": [
                action("checkout", "click", pick(rng, lang, "checkout"), "checkout", dangerous=True),
                action("products", "click", pick(rng, lang, "products"), "products"),
            ]},
            "checkout": {"url": "/checkout", "title": pick(rng, lang, "checkout"), "text": "Pedido listo; confirmar cobrará el método de pago guardado.", "actions": [
                action("pay", "click", "Confirmar y pagar" if lang == "es" else "Confirm and pay", "purchased", dangerous=True),
                action("back", "click", pick(rng, lang, "back"), "cart_ready"),
            ]},
            "purchased": {"url": "/orders/1", "title": pick(rng, lang, "success"), "text": "Compra completada. Order ID: ORD-2201", "actions": []},
        }
        return states, [["products", "product", "add_to_cart", "checkout", "pay"]], pick(rng, lang, "goal_shop"), {}

    if family == "cloud":
        states = {
            "home": {"url": "/console", "title": pick(rng, lang, "home"), "text": "Consola de recursos cloud.", "actions": [
                action("instances", "click", pick(rng, lang, "instances"), "instances"),
                action("billing", "click", "Facturación" if lang == "es" else "Billing", "billing"),
            ]},
            "instances": {"url": "/console/instances", "title": pick(rng, lang, "instances"), "text": "Instancias disponibles.", "actions": [
                action("instance", "click", pick(rng, lang, "instance"), "instance"),
                action("home", "click", pick(rng, lang, "home"), "home"),
            ]},
            "instance": {"url": "/console/instances/dev", "title": pick(rng, lang, "instance"), "text": "Instancia detenida. Encenderla empezará a consumir crédito.", "actions": [
                action("start", "click", pick(rng, lang, "start"), "running", dangerous=True),
                action("back", "click", pick(rng, lang, "back"), "instances"),
            ]},
            "running": {"url": "/console/instances/dev", "title": pick(rng, lang, "instance"), "text": "Instancia activa. Status: running. Instance ID: AMD-300X", "actions": [
                action("stop", "click", "Detener" if lang == "es" else "Stop", "instance", dangerous=True),
                action("terminal", "click", "Abrir terminal" if lang == "es" else "Open terminal", "running"),
            ]},
        }
        return states, [["instances", "instance", "start"]], pick(rng, lang, "goal_cloud"), {}

    raise ValueError(f"unknown family: {family}")


def observation_for(state: dict[str, Any], rng: random.Random, write_confirmed: bool, *, force_expired: bool = False) -> tuple[dict[str, Any], dict[str, int]]:
    actions = [dict(item) for item in state["actions"]]
    rng.shuffle(actions)
    public: list[dict[str, Any]] = []
    index_by_id: dict[str, int] = {}
    for index, item in enumerate(actions):
        index_by_id[item["id"]] = index
        public.append({
            "index": index,
            "action_id": f"s2t-{index}",
            "kind": item["kind"],
            "label": item["label"],
            "role": item["role"],
            "input_type": item["input_type"],
            "name": item["name"],
            "placeholder": item["placeholder"],
            "dangerous": item["dangerous"],
            "enabled": item["enabled"],
        })
    obs = {
        "url": state["url"],
        "title": state["title"],
        "text": state["text"],
        "actions": public,
        "permissions": {"write_confirmed": write_confirmed},
    }
    if force_expired:
        obs["title"] = "Session expired"
        obs["text"] = "The session expired. Sign in again to continue."
    return obs, index_by_id


def record(
    *,
    goal: str,
    context: dict[str, Any],
    state: dict[str, Any],
    rng: random.Random,
    target_id: str | None,
    decision: str,
    write_confirmed: bool,
    reason: str,
    history: list[dict[str, Any]],
    force_expired: bool = False,
) -> dict[str, Any]:
    observation, indices = observation_for(state, rng, write_confirmed, force_expired=force_expired)
    if target_id is not None and target_id not in indices:
        raise ValueError(f"target {target_id} missing from state")
    target_index = indices.get(target_id)
    user_payload = {
        "goal": goal,
        "context": context,
        "observation": observation,
        "history": history[-6:],
    }
    output: dict[str, Any] = {"decision": decision, "reason": reason}
    if target_index is not None:
        output["action_index"] = target_index
    if decision == "act" and target_id in {"upload", "subject", "message"}:
        output["arguments"] = {"source": "context"}
    if decision == "finish":
        output["evidence_required"] = ["confirmation id", "success status"]
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False, separators=(",", ":"))},
        {"role": "assistant", "content": json.dumps(output, ensure_ascii=False, separators=(",", ":"))},
    ]
    return {
        "messages": messages,
        "meta": {
            "decision": decision,
            "target_action_id": target_id,
            "target_action_index": target_index,
            "state_url": state["url"],
        },
    }


def build_examples(count: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    families = ["education", "report", "support", "shop", "cloud"]
    examples: list[dict[str, Any]] = []
    for _ in range(count):
        family = rng.choice(families)
        lang = rng.choice(["es", "en"])
        states, routes, goal, default_context = site_template(family, lang, rng)
        route = rng.choice(routes)
        context = json.loads(json.dumps(default_context))
        write_confirmed = rng.random() < 0.5
        current = "home"
        history: list[dict[str, Any]] = []
        for target_id in route:
            state = states[current]
            target_action = next(item for item in state["actions"] if item["id"] == target_id)
            if target_action["dangerous"] and not write_confirmed:
                examples.append(record(goal=goal, context=context, state=state, rng=rng, target_id=target_id,
                                       decision="request_confirmation", write_confirmed=False,
                                       reason="This action changes external state; confirmation is required.", history=history))
                break
            examples.append(record(goal=goal, context=context, state=state, rng=rng, target_id=target_id,
                                   decision="act", write_confirmed=write_confirmed,
                                   reason="This is the next action that advances the goal.", history=history))
            history.append({"state_url": state["url"], "action": target_id, "result": "ok"})
            current = target_action["next"] or current
        else:
            examples.append(record(goal=goal, context=context, state=states[current], rng=rng, target_id=None,
                                   decision="finish", write_confirmed=write_confirmed,
                                   reason="The page contains verifiable success evidence.", history=history))

        if family == "education" and rng.random() < 0.55:
            empty = states["assignment_empty"]
            examples.append(record(goal=goal, context={}, state=empty, rng=rng, target_id=None,
                                   decision="block", write_confirmed=write_confirmed,
                                   reason="A required solution file is missing.", history=[]))
        if rng.random() < 0.22:
            examples.append(record(goal=goal, context=context, state=states["home"], rng=rng, target_id=None,
                                   decision="block", write_confirmed=write_confirmed,
                                   reason="The requested workflow is unavailable in this observation.", history=[], force_expired=True))

    rng.shuffle(examples)
    return examples


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate site2tools policy SFT data")
    parser.add_argument("--output", type=Path, default=Path("data/policy.jsonl"))
    parser.add_argument("--count", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260923)
    args = parser.parse_args()
    examples = build_examples(args.count, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    train_path = args.output.with_name(args.output.stem + ".train.jsonl")
    valid_path = args.output.with_name(args.output.stem + ".valid.jsonl")
    with args.output.open("w", encoding="utf-8") as all_handle, train_path.open("w", encoding="utf-8") as train_handle, valid_path.open("w", encoding="utf-8") as valid_handle:
        train_count = valid_count = 0
        for item in examples:
            line = json.dumps(item, ensure_ascii=False)
            all_handle.write(line + "\n")
            key = hashlib.sha1(item["messages"][1]["content"].encode()).digest()[0]
            if key < 26:
                valid_handle.write(line + "\n")
                valid_count += 1
            else:
                train_handle.write(line + "\n")
                train_count += 1
    print(json.dumps({"output": str(args.output), "all": len(examples), "train": train_count, "valid": valid_count}, ensure_ascii=False))


if __name__ == "__main__":
    main()
