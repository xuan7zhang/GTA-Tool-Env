"""RaTA-Set-PORTS Stage 1.2 -- normalized tool specifications.

Authored once from the executors' behaviour in tgb/tools.py and tgb/tools_v2.py
and reviewed by hand, then frozen. A spec states what the tool does and what it
cannot do; it never states which task family it belongs to, and no spec
mentions a unit name, currency or document type that appears in any question.

`limitations` is not decoration. Every v2 executor, called on input it was not
designed for, returns a confidently formatted wrong number rather than an
error, and that is the property the whole benchmark turns on -- so each spec
says so in its own words.

    python -m tgb.rata.tool_specs        # writes tool_specs/tool_specs.json
"""
import json
import os

OUT = "/datasets/omni_pretraining/gta2/results/taco/tgb2/rata_set_ports"

SPECS = [
    dict(tool_name="OCR", accepted_inputs=["image"],
         produced_outputs=["exact text", "numbers"],
         capabilities=["transcribe text printed in an image",
                       "recover exact numeric values shown in a document",
                       "preserve the layout order of the printed lines"],
         limitations=["performs no arithmetic",
                      "returns whatever is printed, including labels that are "
                      "not the quantity being asked for"]),
    dict(tool_name="CountGivenObject", accepted_inputs=["image", "text"],
         produced_outputs=["integer count"],
         capabilities=["count how many instances of a named object appear in "
                       "an image"],
         limitations=["reports a count even when no such object is present",
                      "cannot read printed values"]),
    dict(tool_name="GoogleSearch", accepted_inputs=["text"],
         produced_outputs=["records", "numbers"],
         capabilities=["retrieve a record that is not present in the input",
                       "return a listed value associated with a named entity"],
         limitations=["returns nothing when the query matches no record",
                      "the value it returns may not be the one the task needs"]),
    dict(tool_name="KnowledgeBase", accepted_inputs=["text"],
         produced_outputs=["records", "numbers"],
         capabilities=["retrieve a record from an internal catalogue",
                       "return a listed value associated with a named entity"],
         limitations=["holds a second, differently priced revision of the same "
                      "records, so its value can contradict an external "
                      "lookup"]),
    dict(tool_name="Calculator", accepted_inputs=["numbers"],
         produced_outputs=["number"],
         capabilities=["evaluate an arithmetic expression",
                       "sum, multiply and take differences of numbers"],
         limitations=["fails when no expression is supplied",
                      "cannot read values off an input"]),
    dict(tool_name="UnitConvert", accepted_inputs=["number"],
         produced_outputs=["number"],
         capabilities=["restate a magnitude in a different system of length "
                       "measurement"],
         limitations=["applied to a quantity that is not a length it still "
                      "reports a converted magnitude, which is wrong"]),
    dict(tool_name="TempConvert", accepted_inputs=["number"],
         produced_outputs=["number"],
         capabilities=["restate a reading on a different temperature scale"],
         limitations=["applied to a quantity that is not a temperature it "
                      "still reports a converted reading, which is wrong"]),
    dict(tool_name="CurrencyConvert", accepted_inputs=["number"],
         produced_outputs=["number"],
         capabilities=["restate a monetary amount in another currency at a "
                       "supplied rate"],
         limitations=["with no rate supplied it guesses one and reports a "
                      "total-shaped amount that is not the total"]),
    dict(tool_name="DurationCalc", accepted_inputs=["numbers"],
         produced_outputs=["number"],
         capabilities=["compute the elapsed time between two clock readings",
                       "restate a time span as a count of minutes"],
         limitations=["given figures that are not clock readings it still "
                      "reports an elapsed time"]),
    dict(tool_name="Solver", accepted_inputs=["numbers"],
         produced_outputs=["number"],
         capabilities=["recover an unknown from a linear relation it satisfies",
                       "solve for a variable given the coefficients"],
         limitations=["given three unrelated figures it still reports a root"]),
    dict(tool_name="ImageDescription", accepted_inputs=["image"],
         produced_outputs=["free text"],
         capabilities=["describe what an image depicts in general terms"],
         limitations=["does not recover exact printed values"]),
    dict(tool_name="TextToBbox", accepted_inputs=["image", "text"],
         produced_outputs=["coordinates"],
         capabilities=["locate a described object within an image"],
         limitations=["returns only coordinates, no content"]),
    dict(tool_name="Summarize", accepted_inputs=["text"],
         produced_outputs=["free text"],
         capabilities=["restate a passage more briefly"],
         limitations=["rounds and aggregates figures, so its headline number "
                      "is not any figure actually present"]),
    dict(tool_name="Translate", accepted_inputs=["text"],
         produced_outputs=["free text"],
         capabilities=["rewrite a passage in another language"],
         limitations=["perturbs numerals while rewriting"]),
    dict(tool_name="Barcode", accepted_inputs=["image"],
         produced_outputs=["identifier string"],
         capabilities=["decode a barcode into its identifier"],
         limitations=["the identifier encodes nothing about the quantities "
                      "printed on the document"]),
    dict(tool_name="TextToImage", accepted_inputs=["text"],
         produced_outputs=["image"],
         capabilities=["generate an image from a description"],
         limitations=["produces no readable value"]),
    dict(tool_name="DrawBox", accepted_inputs=["image"],
         produced_outputs=["image"],
         capabilities=["draw a rectangle onto an image"],
         limitations=["produces no readable value"]),
    dict(tool_name="AddText", accepted_inputs=["image", "text"],
         produced_outputs=["image"],
         capabilities=["write a caption onto an image"],
         limitations=["produces no readable value"]),
    dict(tool_name="ImageStylization", accepted_inputs=["image"],
         produced_outputs=["image"],
         capabilities=["restyle an image"],
         limitations=["produces no readable value"]),
    dict(tool_name="Plot", accepted_inputs=["text"],
         produced_outputs=["image"],
         capabilities=["render a chart from data"],
         limitations=["produces no readable value"]),
]

BY_NAME = {s["tool_name"]: s for s in SPECS}


def doc_text(s, with_name=True):
    """The string the retriever embeds for a tool."""
    parts = ([f"tool: {s['tool_name']}"] if with_name else []) + [
        "accepts: " + ", ".join(s["accepted_inputs"]),
        "produces: " + ", ".join(s["produced_outputs"]),
        "can: " + "; ".join(s["capabilities"]),
        "cannot: " + "; ".join(s["limitations"])]
    return " ; ".join(parts)


if __name__ == "__main__":
    os.makedirs(f"{OUT}/tool_specs", exist_ok=True)
    json.dump(SPECS, open(f"{OUT}/tool_specs/tool_specs.json", "w"), indent=1)
    print(f"wrote {len(SPECS)} specs")
    for s in SPECS[:3]:
        print(" ", doc_text(s)[:150])
