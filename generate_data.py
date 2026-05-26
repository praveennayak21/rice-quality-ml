"""
Generate synthetic Rice Quality dataset for ML training
Author: Praveen Nagaraj Nayak
"""

import pandas as pd
import numpy as np
import os

def create_dataset(n=500, filepath="data/rice_quality.csv"):
    os.makedirs("data", exist_ok=True)
    np.random.seed(42)

    grades, rows = ["Grade A", "Grade B", "Grade C"], []

    specs = {
        "Grade A": dict(length=(7.5,0.3), width=(3.5,0.2), area=(26,2),
                        perim=(22,1.5), round=(0.85,0.04), ar=(2.1,0.1), color=(0.90,0.04)),
        "Grade B": dict(length=(6.8,0.4), width=(3.0,0.25), area=(21,2.5),
                        perim=(19,1.8), round=(0.76,0.05), ar=(2.3,0.15), color=(0.75,0.05)),
        "Grade C": dict(length=(5.9,0.5), width=(2.5,0.3), area=(16,3),
                        perim=(16,2),   round=(0.65,0.06), ar=(2.6,0.2),  color=(0.60,0.07)),
    }

    per_grade = n // 3
    for grade in grades:
        s = specs[grade]
        for _ in range(per_grade):
            rows.append({
                "Length":       round(np.random.normal(s["length"][0], s["length"][1]), 3),
                "Width":        round(np.random.normal(s["width"][0],  s["width"][1]),  3),
                "Area":         round(np.random.normal(s["area"][0],   s["area"][1]),   3),
                "Perimeter":    round(np.random.normal(s["perim"][0],  s["perim"][1]),  3),
                "Roundness":    round(np.clip(np.random.normal(s["round"][0], s["round"][1]), 0, 1), 4),
                "Aspect_Ratio": round(np.random.normal(s["ar"][0],    s["ar"][1]),     4),
                "Color_Score":  round(np.clip(np.random.normal(s["color"][0], s["color"][1]), 0, 1), 4),
                "Grade":        grade,
            })

    df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
    df.to_csv(filepath, index=False)
    print(f"✅ Dataset created: {filepath} ({len(df)} rows)")
    return filepath

if __name__ == "__main__":
    create_dataset()
