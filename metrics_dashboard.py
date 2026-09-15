import gradio as gr
import pandas as pd

def create_dashboard():
    # Define the research data
    results_data = pd.DataFrame({
        "Evaluation Metric": [
            "Object Detection (mAP @ 0.50:0.95)", 
            "Pose Estimation (OKS AP)", 
            "Image Captioning (CIDEr)", 
            "Image Captioning (BLEU-4)"
        ],
        "Track A (Monolithic)": ["0.037 (3.7%)", "0.001 (0.1%)", "0.462", "0.186"],
        "Track B (Modular)": ["0.220 (22.0%)", "0.450 (45.0%)", "0.765", "0.242"],
        "Capacity Bottleneck Penalty": ["-83.1%", "-99.7%", "-39.6%", "-23.1%"]
    })

    with gr.Blocks(theme=gr.themes.Base()) as dashboard:
        gr.Markdown("# 📊 Research Results: The Capacity Bottleneck on Edge Devices")
        gr.Markdown("### Experimental Data: Track A (Monolithic) vs. Track B (Modular Pipelines)")
        
        # The Data Table
        gr.Dataframe(
            value=results_data,
            headers=list(results_data.columns),
            interactive=False,
            row_count=4,
            col_count=4,
        )
        
        gr.Markdown("---")
        gr.Markdown("## 📖 Metrics Dictionary & Justifications")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("""
                ### 1. mAP (Mean Average Precision)
                **Used For:** Object Detection (Bounding Boxes).
                
                **What it means:** mAP calculates how tightly the predicted bounding box overlaps the ground truth box across multiple thresholds (Intersection over Union). A score of 1.0 means perfect alignment.
                
                **Why we used it:** Simple pixel-distance is flawed because large bounding boxes inherently have larger pixel errors than small boxes. mAP normalizes this, punishing the model equally for loose boxes regardless of the person's size in the frame.
                """)
                
            with gr.Column():
                gr.Markdown("""
                ### 2. OKS (Object Keypoint Similarity)
                **Used For:** Human Pose Estimation (Skeletons).
                
                **What it means:** OKS measures the distance between the predicted joint and the actual joint, but crucially, it scales the penalty based on the size of the person.
                
                **Why we used it:** A 10-pixel error on a massive, close-up human face is acceptable. A 10-pixel error on a tiny human in the background means the joint is floating in mid-air. OKS accounts for scale, making it the academic gold standard for human pose tracking.
                """)

        with gr.Row():
            with gr.Column():
                gr.Markdown("""
                ### 3. CIDEr (Consensus-based Image Description Evaluation)
                **Used For:** Image Captioning (Semantic Value).
                
                **What it means:** CIDEr measures human consensus using TF-IDF (Term Frequency-Inverse Document Frequency). It rewards the model for using unique, descriptive words and ignores common "stop words" like *a*, *the*, or *is*.
                
                **Why we used it:** If the model guesses "A thing is on a thing," it technically used correct English. CIDEr prevents the model from cheating by heavily rewarding specific words like "skateboard" or "tennis racket" that uniquely describe the scene.
                """)
                
            with gr.Column():
                gr.Markdown("""
                ### 4. BLEU-4 (Bilingual Evaluation Understudy)
                **Used For:** Image Captioning (Grammatical Fluency).
                
                **What it means:** Originally built for language translation, BLEU counts how many 4-word sequences (n-grams) in the model's prediction directly match the human-written ground truth caption.
                
                **Why we used it:** While CIDEr measures *what* the model is saying, BLEU-4 measures *how* it says it. It ensures the neural network is forming structurally sound, grammatically correct English sequences rather than just spitting out a list of random nouns.
                """)
                
    return dashboard

if __name__ == "__main__":
    app = create_dashboard()
    app.launch()