import dspy
import os
import json

from eval_function import table_extraction_metric
    


class TableRow(dspy.Signature):
    """Extract all rows from a table image. Each row has: date, description, location, amount."""
    image: dspy.Image = dspy.InputField(desc="Image of a table with one or more rows")
    rows: list[dict] = dspy.OutputField(
        desc="List of dicts with keys: date (YYYY-MM-DD), description (str), location (str), amount (float)"
    )
    
class TableExtractor(dspy.Module):
    def __init__(self):
        self.extractor = dspy.ChainOfThought(TableRow)

    def forward(self, image_path: str) -> dspy.Prediction:
        image = dspy.Image.from_file(image_path)
        return self.extractor(image=image)

def load_dataset():

    with open("images/dataset.json", "r") as f:
        dataset = json.load(f)
        examples = []
        for row in dataset:
            examples += [
                dspy.Example(image_path=d["image_path"], rows=d["rows"]).with_inputs("image_path")
                for d in row
            ]
        
    return examples


if __name__ == "__main__":
    dataset = load_dataset()
    split = int(len(dataset) * 0.8)
    trainset, valset = dataset[:split], dataset[split:]

    # Compile with GEPA
    program = TableExtractor()


    llm =dspy.LM(
        os.getenv("MODEL_NAME"), 
        api_key=os.getenv("OPEN_API_KEY"), 
        api_base=os.getenv("OPEN_API_URL")
    )
    
    dspy.configure(
        lm=llm
    )

    reflection_lm = llm

    optimizer = dspy.GEPA(
        metric=table_extraction_metric,
        reflection_lm=reflection_lm,
        auto="medium",                        # "light" | "medium" | "heavy"
        reflection_minibatch_size=4,          # examples per reflection batch
        candidate_selection_strategy="pareto",
        use_merge=True,
    )

    optimized_program = optimizer.compile(
        student=program,
        trainset=trainset,
        valset=valset,
    )

    optimized_program.save("table_extractor_optimized.json")

    # ── Evaluate ──
    evaluator = dspy.Evaluate(
        devset=valset,
        metric=table_extraction_metric,
        num_threads=4,
        display_progress=True,
    )
    evaluator(optimized_program)