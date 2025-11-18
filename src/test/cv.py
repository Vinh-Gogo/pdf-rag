system_prompt = """You are an IT HR Recruiter AI Agent.
Your task is to evaluate CVs using ONLY the information explicitly provided by the user.
No assumptions. No hallucinations. No invented details.

Your evaluation MUST follow the EXACT output structure below, with no additional sections:
1. Summary
2. Strengths
3. Weaknesses
4. Role Fit Score (1–4)
5. Recommended Roles
6. Actionable Suggestions

GENERAL RULES:
- Use bullet points.
- Be concise, objective, and professional.
- Do not use emotional language.
- Do not provide long paragraphs.
- If any information is missing or unclear, write: "Not provided."

STRICT INFORMATION RULES:
- Do NOT infer skills, roles, or experience that the CV does not state.
- Do NOT assume seniority unless explicitly stated.
- Do NOT judge personal traits (age, gender, personality, etc.).
- Evaluate ONLY technical and professional information.
- If the user gives both a CV and a Job Description, analyze both but do not add anything external.

EVALUATION GUIDELINES:
When generating Strengths, Weaknesses, and Fit Score, consider ONLY:
- Programming languages mentioned
- Frameworks, tools, and technologies listed
- Relevant work experience duration and responsibilities
- Education background (if provided)
- Technical projects and achievements
- Certifications (if provided)
- Project complexity and relevance to common IT roles

FIT SCORE DEFINITION (MANDATORY):
1 = Not suitable (missing most required skills or unrelated background)
2 = Partially suitable (some relevant skills but significant gaps)
3 = Suitable (meets most skill requirements with minor gaps)
4 = Highly suitable (strong match for the role with clear supporting evidence)

RECOMMENDED ROLES:
- Suggest 2–5 IT roles based ONLY on the skills shown in the CV.
- Do NOT recommend roles the candidate clearly cannot perform.

ACTIONABLE SUGGESTIONS:
- Suggest realistic improvements (skills to learn, specific tools to study, project types to build).
- Must be directly connected to weaknesses found in the CV.
- Keep them practical, short, and job-market relevant.

FORMATTING RULES:
- Use concise bullet points.
- No emojis.
- No unnecessary explanation.
- Do not break the required structure.

ERROR HANDLING:
If the user provides:
- An incomplete CV → evaluate only what is provided and mark missing data as "Not provided."
- A non-IT CV → still evaluate but maintain same rules.
- Conflicting information → highlight it objectively in Weaknesses.

Your response must ALWAYS follow the exact 6-section structure mentioned above without deviation.
"""

nmy_cv = '''
![](_page_0_Picture_0.jpeg)

# **Le Quang Vinh**

*AI Engineer* | *Prompt Engineer* | *Software Engineer*

Phone-Alt *[+84 985 189 541](tel:+84985189541)* | Envelope *[lea26462@gmail.com](mailto:lea26462@gmail.com)* | Github *[github.com/Vinh-Gogo](https://github.com/Vinh-Gogo )* | BIRTHDAY-CAKE *Feb 23, 2001*

*From:* Map-marker-alt *An Giang Province* | *Work:* Map-marker-alt *Ho Chi Minh City*

# **Career Objective**

Leverage CNNs and LLMs to automate workflows and enhance operational efficiency. Seek real-world AI-driven projects where I can deliver measurable impact while maximizing the practical advantages of AI technologies.

# **Education**

Industrial University of Ho Chi Minh City **2021 – 2025**

B.Sc. Computer Science

# **Work Experience**

**Backend Contractor 05/2025 – 07/2025**

*Office of Investment & Planning, IUH* LINK [Deployed: pkhdtiuh.com](https://pkhdtiuh.com)

# **Projects**

*SuperNova MCP RAG - Team: 2 people* **08/2025 – Current**

# **SUPER NOVA RAG: Vietnamese Multi-Agent System** (Python)

- *A hierarchical system with one AI Executive Director overseeing specialized AI agents (RAG, Database, Planning, Translation). The director supervises their work, verifies results, and responds to user queries within 15 seconds. The system supports multiple languages.*
- *Ability to learn PDF documents over 100 pages and provide answers based on the supplied knowledge.*
- *Ensures strict control over humanity, accuracy, and user satisfaction.*

**Use Cases:** Supports consulting and conveying the values or products you have to users.

**AI Chat FB:** LINK [PyPy - Automatics Agent Mini](https://www.facebook.com/profile.php?id=61580840842607&mibextid=ZbWKwL)

**Backend Framework & APIs:** OpenAI, LangChain, Langgraph, Docker / APIs Services, Models Open Source.

**Benchmark:** 3 200 queries, RAG hybrid accuracy: **95 ± 1%** Hit@1, **99 ± 0.9%** Hit@5.

*Depth Estimation (Thesis & Researching) - Team: 3 people.* **02/2025 – Current**

# **CNN-Based Image Depth Estimation Methods** (Tensorflow)

*- Computer Vision. The eyes of the system are positioned above – they serve as the vision that perceives and understands how the human world operates.*

**Use Cases:** 3D Reconstruction (Point Cloud), Object Segmentation, Industrial Applications, Pose Estimation.

**Source Part 1:** Github [github.com/Vinh-Gogo/depth-estimation](https://github.com/Vinh-Gogo/depth-estimation)

#### **Core Technologies:**

- Convolutional Autoencoder (CAE), ResNet, U-Net, U-Net combining ResNet & DenseNet, U-Net SE ResNet-Dense with Squeeze-and-Excitation, Attention block, Transformers.

**Evaluate:** Image Similarity & Distance Metrics. Balance between **accuracy** and **efficiency.**

**Dataset:** LineMOD Dataset (Available on Google Drive, GitHub and BOP website).

**Results Test (1 200 images):** Accuracy: 90%, Cosine Similarity: 97%.

# **Soft Skills**

Weekly tech-share (30+ session), Communication, Problem-solving, Logical thinking, Time management.
'''



from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, os
from pathlib import Path
import re

# ⚙️ GPU config
torch.set_float32_matmul_precision("high")

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
# Load model directly

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B").to(device)
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": f"Evaluate the following CV:\n\n{nmy_cv}"}
]
inputs = tokenizer.apply_chat_template(
	messages,
	add_generation_prompt=True,
	tokenize=True,
	return_dict=True,
	return_tensors="pt",
	enable_thinking=False
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=32768)
result = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:])

# Define a path to save the result
output_path = Path(r"src/test/cv_evaluation_result_vi.txt")

# Save the result to a text file
with open(output_path, "w", encoding="utf-8") as f:
    f.write(result)

print(f"Result saved to {output_path.resolve()}")