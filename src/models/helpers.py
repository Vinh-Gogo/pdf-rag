import numpy as np
# from __future__ import annotations

# import os
# import re
# import unicodedata
# from pathlib import Path
# from typing import List, Dict, Any, Optional, Union, Protocol
# from underthesea import word_tokenize, pos_tag

# import pandas as pd
# import requests
# import cohere
# from langchain.text_splitter import CharacterTextSplitter
# from langchain_community.document_loaders import CSVLoader
# from langchain_neo4j import Neo4jVector
# from pathlib import Path
# from typing import Optional, Union
# # VietnameseToneNormalization.md
# # https://github.com/VinAIResearch/BARTpho/blob/main/VietnameseToneNormalization.md


def cosine_similarity(vec1, vec2):
    """Tính độ tương đồng cosine giữa 2 vector"""
    vec1_norm = vec1 / np.linalg.norm(vec1)
    vec2_norm = vec2 / np.linalg.norm(vec2)
    return float(np.dot(vec1_norm, vec2_norm))