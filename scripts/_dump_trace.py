#!/usr/bin/env python
"""Dump full LLM interaction traces for case study comparison."""
import json, sys, textwrap

datasets = sys.argv[1:] if len(sys.argv) > 1 else ["DermaMNIST", "ISIC2019"]

STEP_NAMES = {
    "v9_interpret": "Phase 2: INTERPRET (LLM #1 — blind, multimodal)",
    "v9_analyze": "Phase 3: ANALYZE (LLM #2 — hypothesis, no benchmarks)",
    "v9_act": "Phase 4: ACT (LLM #3 — reconcile with benchmarks + memory)",
    "v9_reflect": "Phase 6: REFLECT (LLM #4 — evaluate quality + write memory)",
}

for ds in datasets:
    cs = json.load(open("results/case_studies/%s_explicit_case_study.json" % ds))
    d = cs["decisions"]

    print("\n" + "=" * 100)
    print("  %s — FULL TRACE" % ds)
    print("=" * 100)

    # Phase 1: OBSERVE (deterministic — no LLM)
    print("\n" + "-" * 80)
    print("  Phase 1: OBSERVE (deterministic — no LLM call)")
    print("-" * 80)
    # Extract tool info from decisions
    print("  Tools executed: image_loader, image_analyzer, compute_ph, topological_features, betti_ratios")
    print("  Filtration: sublevel (always fixed in v9)")

    # Print each LLM interaction
    interactions = cs.get("llm_interactions", [])
    for ix in interactions:
        step = ix.get("step", "?")
        prompt = ix.get("prompt", "")
        response = ix.get("response", "")

        print("\n" + "-" * 80)
        print("  %s" % STEP_NAMES.get(step, step))
        print("-" * 80)

        print("\n  >>> PROMPT (%d chars) >>>" % len(prompt))
        # Print prompt with wrapping
        for line in prompt.split("\n"):
            if line.strip():
                print("  | %s" % line)
            else:
                print("  |")

        print("\n  <<< GPT RESPONSE (%d chars) <<<" % len(response))
        # Try to parse as JSON for pretty printing
        try:
            parsed = json.loads(response.strip())
            print("  %s" % json.dumps(parsed, indent=2, default=str).replace("\n", "\n  "))
        except json.JSONDecodeError:
            # Try markdown code block
            if "```json" in response:
                start = response.index("```json") + 7
                end = response.index("```", start)
                try:
                    parsed = json.loads(response[start:end].strip())
                    print("  %s" % json.dumps(parsed, indent=2, default=str).replace("\n", "\n  "))
                except:
                    for line in response.split("\n"):
                        print("  | %s" % line)
            elif "{" in response:
                bstart = response.find("{")
                bend = response.rfind("}")
                try:
                    parsed = json.loads(response[bstart:bend+1])
                    print("  %s" % json.dumps(parsed, indent=2, default=str).replace("\n", "\n  "))
                except:
                    for line in response.split("\n"):
                        print("  | %s" % line)
            else:
                for line in response.split("\n"):
                    print("  | %s" % line)

    # Phase 5: EXTRACT (deterministic)
    print("\n" + "-" * 80)
    print("  Phase 5: EXTRACT (deterministic — no LLM call)")
    print("-" * 80)
    act = d.get("v9_act_decision", {})
    print("  Descriptor: %s" % act.get("final_descriptor", "?"))
    print("  Params: %s" % act.get("final_params", {}))
    print("  Color mode: %s" % act.get("color_mode", "?"))

    # Summary
    print("\n" + "-" * 80)
    print("  SUMMARY")
    print("-" * 80)
    interp = d.get("v9_interpret", {})
    hyp = d.get("v9_hypothesis", {})
    refl = d.get("v9_reflect", {})
    print("  OT guess: %s -> reconciled: %s" % (
        interp.get("object_type_guess", "?"), hyp.get("object_type_reconciled", "?")))
    print("  Hypothesis: %s (confidence=%s)" % (
        hyp.get("descriptor_hypothesis", "?"), hyp.get("confidence", "?")))
    print("  Final: %s (stance=%s)" % (
        act.get("final_descriptor", "?"), act.get("_inferred_stance", "?")))
    print("  Quality: %s" % refl.get("quality_assessment", {}).get("overall_quality", "?"))
    print("  Decision: %s" % refl.get("decision", "?"))
    exp = refl.get("experience_entry", {})
    print("  LTM lesson: %s" % exp.get("lesson", "?"))
    print("  Would choose again: %s" % exp.get("would_choose_again", "?"))
