from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

from generate_dataset import SYSTEM_PROMPT, action, pick, record


def tr(lang: str, es: str, en: str) -> str:
    return es if lang == "es" else en


def state(url: str, title: str, text: str, actions: list[dict[str, Any]]) -> dict[str, Any]:
    return {"url": url, "title": title, "text": text, "actions": actions}


def long_template(
    family: str, lang: str, rng: random.Random
) -> tuple[dict[str, dict[str, Any]], list[str], str, dict[str, Any]]:
    """Return a 7-10 action workflow with distractors and a terminal proof."""
    if family == "education":
        states = {
            "home": state("/home", pick(rng, lang, "home"), tr(lang, "Portal educativo con buscador.", "Education portal with search."), [
                action("search", "click", tr(lang, "Buscar cursos", "Search courses"), "course_search"),
                action("courses", "click", pick(rng, lang, "courses"), "courses"),
                action("profile", "click", tr(lang, "Perfil", "Profile"), "home"),
            ]),
            "courses": state("/courses", pick(rng, lang, "courses"), tr(lang, "Catálogo de cursos.", "Course catalog."), [
                action("algebra", "click", pick(rng, lang, "algebra"), "course"),
                action("search", "click", tr(lang, "Buscar cursos", "Search courses"), "course_search"),
                action("home", "click", pick(rng, lang, "home"), "home"),
            ]),
            "course_search": state("/courses/search", tr(lang, "Buscar cursos", "Search courses"), tr(lang, "Escribe el curso que necesitas.", "Enter the course you need."), [
                action("query", "input", tr(lang, "Nombre del curso", "Course name"), "course_results", input_type="text", name="query"),
                action("home", "click", pick(rng, lang, "home"), "home"),
            ]),
            "course_results": state("/courses/search/results", tr(lang, "Resultados", "Results"), tr(lang, "Cursos encontrados.", "Courses found."), [
                action("algebra", "click", pick(rng, lang, "algebra"), "course"),
                action("search", "click", tr(lang, "Nueva búsqueda", "New search"), "course_search"),
            ]),
            "course": state("/courses/algebra", pick(rng, lang, "algebra"), tr(lang, "Curso seleccionado.", "Selected course."), [
                action("activities", "click", tr(lang, "Actividades", "Activities"), "activities"),
                action("courses", "click", pick(rng, lang, "courses"), "courses"),
            ]),
            "activities": state("/courses/algebra/activities", tr(lang, "Actividades", "Activities"), tr(lang, "Hay una tarea pendiente.", "There is a pending assignment."), [
                action("assignment", "click", pick(rng, lang, "assignment"), "assignment_empty"),
                action("course", "click", pick(rng, lang, "algebra"), "course"),
            ]),
            "assignment_empty": state("/assignments/algebra", pick(rng, lang, "assignment"), tr(lang, "Sube tu solución en PDF y entrégala.", "Upload your PDF solution and submit it."), [
                action("upload", "input", pick(rng, lang, "upload"), "assignment_ready", input_type="file", name="solution", placeholder="PDF"),
                action("submit", "click", pick(rng, lang, "submit"), "submitted", dangerous=True),
                action("back", "click", pick(rng, lang, "back"), "activities"),
            ]),
            "assignment_ready": state("/assignments/algebra", pick(rng, lang, "assignment"), tr(lang, "Archivo seleccionado; entregar cambia el estado.", "File selected; submit changes external state."), [
                action("submit", "click", pick(rng, lang, "submit"), "submitted", dangerous=True),
                action("back", "click", pick(rng, lang, "back"), "activities"),
            ]),
            "submitted": state("/assignments/algebra/submitted", pick(rng, lang, "success"), "Tarea entregada. Confirmation ID: EDU-LONG-83921", []),
        }
        return states, ["search", "query", "algebra", "activities", "assignment", "upload", "submit"], pick(rng, lang, "goal_edu"), {"file": "solution.pdf", "inputs": {"query": tr(lang, "Álgebra", "Algebra")}}

    if family == "report":
        states = {
            "home": state("/dashboard", pick(rng, lang, "home"), tr(lang, "Panel con informes.", "Dashboard with reports."), [
                action("reports", "click", pick(rng, lang, "reports"), "reports"),
                action("profile", "click", tr(lang, "Perfil", "Profile"), "home"),
            ]),
            "reports": state("/reports", pick(rng, lang, "reports"), tr(lang, "Elige un periodo para el informe.", "Choose a report period."), [
                action("filters", "click", tr(lang, "Configurar filtros", "Configure filters"), "filters"),
                action("home", "click", pick(rng, lang, "home"), "home"),
            ]),
            "filters": state("/reports/filters", tr(lang, "Filtros", "Filters"), tr(lang, "Configura el periodo.", "Configure the period."), [
                action("period", "input", tr(lang, "Periodo", "Period"), "period_set", input_type="text", name="period"),
                action("back", "click", pick(rng, lang, "back"), "reports"),
            ]),
            "period_set": state("/reports/filters", tr(lang, "Filtros", "Filters"), tr(lang, "Periodo elegido; aplica los filtros.", "Period selected; apply filters."), [
                action("apply", "click", tr(lang, "Aplicar filtros", "Apply filters"), "report_results"),
                action("period", "input", tr(lang, "Periodo", "Period"), "period_set", input_type="text", name="period"),
            ]),
            "report_results": state("/reports/results", tr(lang, "Resultados", "Results"), tr(lang, "Informes disponibles.", "Reports available."), [
                action("monthly", "click", pick(rng, lang, "report"), "report"),
                action("filters", "click", tr(lang, "Cambiar filtros", "Change filters"), "filters"),
            ]),
            "report": state("/reports/monthly", pick(rng, lang, "report"), tr(lang, "Reporte listo para exportar.", "Report ready to export."), [
                action("export", "click", pick(rng, lang, "export"), "exported", dangerous=True),
                action("back", "click", pick(rng, lang, "back"), "report_results"),
            ]),
            "exported": state("/reports/monthly/download", pick(rng, lang, "success"), "Reporte exportado. Download ID: CSV-LONG-4812", []),
        }
        return states, ["reports", "filters", "period", "apply", "monthly", "export"], pick(rng, lang, "goal_report"), {"inputs": {"period": "2026-09"}}

    if family == "support":
        states = {
            "home": state("/dashboard", pick(rng, lang, "home"), tr(lang, "Panel con soporte.", "Dashboard with support."), [
                action("support", "click", pick(rng, lang, "support"), "support"),
                action("profile", "click", tr(lang, "Perfil", "Profile"), "home"),
            ]),
            "support": state("/support", pick(rng, lang, "support"), tr(lang, "Elige una categoría.", "Choose a category."), [
                action("category", "input", tr(lang, "Categoría", "Category"), "category_set", input_type="text", name="category"),
                action("home", "click", pick(rng, lang, "home"), "home"),
            ]),
            "category_set": state("/support", pick(rng, lang, "support"), tr(lang, "Categoría elegida; crea una solicitud.", "Category selected; create a request."), [
                action("new_ticket", "click", pick(rng, lang, "new_ticket"), "ticket_empty"),
                action("category", "input", tr(lang, "Categoría", "Category"), "category_set", input_type="text", name="category"),
            ]),
            "ticket_empty": state("/support/new", pick(rng, lang, "new_ticket"), tr(lang, "Completa asunto y descripción.", "Complete subject and description."), [
                action("subject", "input", pick(rng, lang, "subject"), "ticket_subject", input_type="text", name="subject"),
                action("message", "input", pick(rng, lang, "message"), "ticket_message", input_type="text", name="message"),
                action("send", "click", pick(rng, lang, "send"), "ticket_submitted", dangerous=True, enabled=False),
            ]),
            "ticket_subject": state("/support/new", pick(rng, lang, "new_ticket"), tr(lang, "Asunto listo; falta descripción.", "Subject ready; description is missing."), [
                action("message", "input", pick(rng, lang, "message"), "ticket_message", input_type="text", name="message"),
                action("subject", "input", pick(rng, lang, "subject"), "ticket_subject", input_type="text", name="subject"),
            ]),
            "ticket_message": state("/support/new", pick(rng, lang, "new_ticket"), tr(lang, "Descripción lista; revisa la solicitud.", "Description ready; review the request."), [
                action("review", "click", tr(lang, "Revisar solicitud", "Review request"), "ticket_review"),
                action("message", "input", pick(rng, lang, "message"), "ticket_message", input_type="text", name="message"),
            ]),
            "ticket_review": state("/support/review", tr(lang, "Revisión", "Review"), tr(lang, "Elige prioridad antes de enviar.", "Choose priority before sending."), [
                action("priority", "input", tr(lang, "Prioridad", "Priority"), "ticket_ready", input_type="text", name="priority"),
                action("send", "click", pick(rng, lang, "send"), "ticket_submitted", dangerous=True, enabled=False),
            ]),
            "ticket_ready": state("/support/review", tr(lang, "Revisión", "Review"), tr(lang, "Solicitud lista; enviar crea un ticket externo.", "Request ready; sending creates an external ticket."), [
                action("send", "click", pick(rng, lang, "send"), "ticket_submitted", dangerous=True),
                action("priority", "input", tr(lang, "Prioridad", "Priority"), "ticket_ready", input_type="text", name="priority"),
            ]),
            "ticket_submitted": state("/support/ticket/1001", pick(rng, lang, "success"), "Solicitud creada. Ticket ID: SUP-LONG-1001", []),
        }
        context = {"inputs": {"category": tr(lang, "Curso", "Course"), "subject": tr(lang, "No puedo abrir el curso", "I cannot open the course"), "message": tr(lang, "El enlace devuelve un error.", "The link returns an error."), "priority": tr(lang, "normal", "normal")}}
        return states, ["support", "category", "new_ticket", "subject", "message", "review", "priority", "send"], pick(rng, lang, "goal_ticket"), context

    raise ValueError(f"unsupported long-horizon family: {family}")


def build_examples(count: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    families = ["education", "report", "support"]
    examples: list[dict[str, Any]] = []
    for scenario in range(count):
        family = rng.choice(families)
        lang = rng.choice(["es", "en"])
        states, route, goal, context = long_template(family, lang, rng)
        write_confirmed = rng.random() < 0.5
        current = "home"
        history: list[dict[str, Any]] = []
        for step, target_id in enumerate(route):
            current_state = states[current]
            target = next(item for item in current_state["actions"] if item["id"] == target_id)
            decision = "request_confirmation" if target["dangerous"] and not write_confirmed else "act"
            item = record(
                goal=goal,
                context=context,
                state=current_state,
                rng=rng,
                target_id=target_id,
                decision=decision,
                write_confirmed=write_confirmed,
                reason=(
                    "This action changes external state; confirmation is required."
                    if decision == "request_confirmation"
                    else "This is the next action that advances the goal."
                ),
                history=history,
            )
            item["meta"].update({
                "kind": "trajectory",
                "family": family,
                "lang": lang,
                "scenario": scenario,
                "step": step,
                "route_length": len(route),
                "write_confirmed": write_confirmed,
            })
            examples.append(item)
            if decision == "request_confirmation":
                break
            history.append({"state_url": current_state["url"], "action": target_id, "result": "ok"})
            current = target["next"] or current
        else:
            item = record(
                goal=goal,
                context=context,
                state=states[current],
                rng=rng,
                target_id=None,
                decision="finish",
                write_confirmed=write_confirmed,
                reason="The page contains verifiable success evidence.",
                history=history,
            )
            item["meta"].update({
                "kind": "trajectory",
                "family": family,
                "lang": lang,
                "scenario": scenario,
                "step": len(route),
                "route_length": len(route),
                "write_confirmed": write_confirmed,
            })
            examples.append(item)
    rng.shuffle(examples)
    return examples


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate long-horizon site2tools policy data")
    parser.add_argument("--output", type=Path, default=Path("data/policy-long.jsonl"))
    parser.add_argument("--count", type=int, default=2500)
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
            # Keep complete scenarios in one split so validation tests generalization to new episodes.
            if item["meta"]["scenario"] % 10 == 0:
                valid_handle.write(line + "\n")
                valid_count += 1
            else:
                train_handle.write(line + "\n")
                train_count += 1
    print(json.dumps({"output": str(args.output), "all": len(examples), "train": train_count, "valid": valid_count}, ensure_ascii=False))


if __name__ == "__main__":
    main()
