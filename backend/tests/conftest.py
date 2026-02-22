import sys
import os

# Add backend/ to sys.path so that `import ai_generator`, `import vector_store`, etc. work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
