
# from typing import Dict, Any
# import openai
# import os
# from dotenv import load_dotenv
# import sys
# from io import StringIO
# import re
# import traceback


# def remove_backticks(text: str) -> str:
#     """Remove code block backticks from text"""
#     return re.sub(r"```\w*\n?(.*?)\n?```", r"\1", text, flags=re.DOTALL).strip()


# # Load environment variables
# load_dotenv()


# class CodeAgent:
#     def __init__(self):
#         self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
#     def execute(self, task: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
#         """Generate and execute code based on the task"""
        
#         try:
#             # Generate code
#             code = self._generate_code(task, context)
            
#             # Test code execution
#             execution_result = self._test_code(code)
            
#             # Generate documentation
#             documentation = self._generate_documentation(code, task)
            
#             return {
#                 "code": code,
#                 "execution_result": execution_result,
#                 "documentation": documentation,
#                 "language": "python",
#                 "status": "completed"
#             }
        
#         except Exception as e:
#             return {
#                 "error": str(e),
#                 "traceback": traceback.format_exc(),
#                 "status": "failed",
#                 "fallback_code": self._get_fallback_code(task)
#             }
    
#     def _generate_code(self, task: str, context: Dict[str, Any] = None) -> str:
#         """Generate code implementation using OpenAI API"""
        
#         context_info = ""
#         if context and context.get("research"):
#             research_data = context["research"]
#             context_info = f"""
#             Research Context:
#             {research_data.get('synthesis', '')[:1000]}
#             """
        
#         prompt = f"""
#         Generate Python code for this task:
#         Task: {task}
        
#         {context_info}
        
#         Requirements:
#         - Write clean, documented Python code
#         - Include comments explaining key parts
#         - Make it executable and functional
#         - Focus on the core algorithm/implementation
#         - Use standard libraries when needed
#         - Include proper error handling

#         Output:
#         ```python
#         {{code}}
#         ```

#         IMPORTANT: Return only the Python code in backticks, no explanations.
#         """
        
#         try:
#             response = self.client.chat.completions.create(
#                 model="gpt-4.1",
#                 messages=[{"role": "user", "content": prompt}],
#                 max_tokens=1500,
#                 temperature=0.2
#             )
            
#             return remove_backticks(response.choices[0].message.content)
        
#         except Exception as e:
#             print(f"Code generation error: {e}")
#             return self._get_fallback_code(task)
    
#     def _test_code(self, code: str) -> Dict[str, Any]:
#         """Safely test the generated code"""
        
#         try:
#             # Capture output
#             old_stdout = sys.stdout
#             sys.stdout = captured_output = StringIO()
            
#             # Execute code in restricted environment
#             exec_globals = {
#                 '__builtins__': {
#                     'print': print,
#                     'len': len,
#                     'range': range,
#                     'str': str,
#                     'int': int,
#                     'float': float,
#                     'list': list,
#                     'dict': dict,
#                     'tuple': tuple,
#                     'set': set,
#                     'enumerate': enumerate,
#                     'sum': sum,
#                     'zip': zip,
#                     'min': min,
#                     'max': max,
#                     'abs': abs,
#                     'round': round,
#                     'sorted': sorted,
#                     'reversed': reversed
#                 }
#             }
            
#             # Add safe imports
#             try:
#                 import numpy as np
#                 import hashlib
#                 import random
#                 import math
#                 import datetime
#                 exec_globals.update({
#                     'np': np,
#                     'hashlib': hashlib,
#                     'random': random,
#                     'math': math,
#                     'datetime': datetime
#                 })
#             except ImportError:
#                 pass
            
#             exec(code, exec_globals)
            
#             # Restore stdout
#             sys.stdout = old_stdout
#             output = captured_output.getvalue()
            
#             return {
#                 "success": True,
#                 "output": output.strip(),
#                 "error": None
#             }
        
#         except Exception as e:
#             sys.stdout = old_stdout
#             return {
#                 "success": False,
#                 "output": None,
#                 "error": str(e),
#                 "traceback": traceback.format_exc()
#             }
    
#     def _generate_documentation(self, code: str, task: str) -> str:
#         """Generate documentation for the code"""
        
#         prompt = f"""
#         Generate clear documentation for this code:
        
#         Task: {task}
#         Code:
#         {code}
        
#         Provide:
#         1. Brief description of what the code does
#         2. Key functions/classes and their purpose
#         3. How to use/run the code
#         4. Any important notes or limitations
        
#         Keep it concise but informative.
#         """
        
#         try:
#             response = self.client.chat.completions.create(
#                 model="gpt-4.1",
#                 messages=[{"role": "user", "content": prompt}],
#                 max_tokens=500,
#                 temperature=0.3
#             )
            
#             return remove_backticks(response.choices[0].message.content)
        
#         except Exception:
#             return f"Documentation for code implementation of: {task}"
    
#     def _get_fallback_code(self, task: str) -> str:
#         """Provide fallback code when generation fails"""
        
#         # Check for specific task types and provide relevant fallbacks
#         task_lower = task.lower()
        
#         if "quantum" in task_lower and ("algorithm" in task_lower or "cryptography" in task_lower):
#             return '''
# # Quantum-Resistant Algorithm Example - Lattice-based Cryptography
# import hashlib
# import random

# class SimpleLatticeEncryption:
#     """
#     Simplified lattice-based encryption (for demonstration)
#     Real implementations would use more sophisticated mathematical structures
#     """
    
#     def __init__(self, dimension=10, modulus=97):
#         self.n = dimension
#         self.q = modulus
#         self.private_key = self._generate_private_key()
#         self.public_key = self._generate_public_key()
    
#     def _generate_private_key(self):
#         """Generate private key as small random integers"""
#         return [random.randint(-2, 2) for _ in range(self.n)]
    
#     def _generate_public_key(self):
#         """Generate public key using private key"""
#         public = []
#         for i in range(self.n):
#             val = sum(self.private_key[j] * random.randint(1, 10) for j in range(self.n))
#             public.append(val % self.q)
#         return public
    
#     def encrypt(self, message_bit):
#         """Encrypt a single bit"""
#         noise = random.randint(-1, 1)
#         ciphertext = sum(self.public_key[i] * (message_bit + noise) for i in range(self.n)) % self.q
#         return ciphertext
    
#     def demonstrate(self):
#         """Demonstrate the algorithm"""
#         print("Quantum-Resistant Lattice-based Encryption Demo")
#         print(f"Dimension: {self.n}, Modulus: {self.q}")
#         print(f"Private Key: {self.private_key}")
#         print(f"Public Key: {self.public_key}")
        
#         # Encrypt a message
#         message = 1  # Single bit
#         encrypted = self.encrypt(message)
#         print(f"Original message bit: {message}")
#         print(f"Encrypted: {encrypted}")

# # Run demonstration
# if __name__ == "__main__":
#     lattice_crypto = SimpleLatticeEncryption()
#     lattice_crypto.demonstrate()
# '''
        
#         elif "machine learning" in task_lower or "ml" in task_lower:
#             return '''
# # Basic Machine Learning Implementation
# import random
# import math

# class SimpleLinearRegression:
#     """Simple linear regression implementation"""
    
#     def __init__(self):
#         self.slope = 0
#         self.intercept = 0
    
#     def fit(self, x_data, y_data):
#         """Train the model"""
#         n = len(x_data)
#         sum_x = sum(x_data)
#         sum_y = sum(y_data)
#         sum_xy = sum(x * y for x, y in zip(x_data, y_data))
#         sum_x2 = sum(x * x for x in x_data)
        
#         self.slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
#         self.intercept = (sum_y - self.slope * sum_x) / n
    
#     def predict(self, x):
#         """Make prediction"""
#         return self.slope * x + self.intercept
    
#     def demonstrate(self):
#         """Demonstrate the algorithm"""
#         # Sample data
#         x_data = [1, 2, 3, 4, 5]
#         y_data = [2, 4, 6, 8, 10]
        
#         self.fit(x_data, y_data)
        
#         print(f"Linear Regression Model")
#         print(f"Slope: {self.slope:.2f}")
#         print(f"Intercept: {self.intercept:.2f}")
#         print(f"Prediction for x=6: {self.predict(6):.2f}")

# if __name__ == "__main__":
#     model = SimpleLinearRegression()
#     model.demonstrate()
# '''
        
#         elif "sorting" in task_lower or "algorithm" in task_lower:
#             return '''
# # Sorting Algorithm Implementation
# def quick_sort(arr):
#     """Quick sort implementation"""
#     if len(arr) <= 1:
#         return arr
    
#     pivot = arr[len(arr) // 2]
#     left = [x for x in arr if x < pivot]
#     middle = [x for x in arr if x == pivot]
#     right = [x for x in arr if x > pivot]
    
#     return quick_sort(left) + middle + quick_sort(right)

# def demonstrate_sorting():
#     """Demonstrate sorting algorithms"""
#     import random
    
#     # Generate random data
#     data = [random.randint(1, 100) for _ in range(10)]
#     print(f"Original array: {data}")
    
#     # Sort using quick sort
#     sorted_data = quick_sort(data.copy())
#     print(f"Sorted array: {sorted_data}")
    
#     # Verify sorting
#     is_sorted = all(sorted_data[i] <= sorted_data[i+1] for i in range(len(sorted_data)-1))
#     print(f"Correctly sorted: {is_sorted}")

# if __name__ == "__main__":
#     demonstrate_sorting()
# '''
        
#         else:
#             return f'''
# # Generated code for: {task}
# def main():
#     """
#     Implementation for the requested task: {task}
#     """
#     print("Code implementation for: {task}")
    
#     # Basic structure - would be expanded based on specific requirements
#     result = "Task completed successfully"
#     return result

# def demonstrate():
#     """Demonstrate the implementation"""
#     result = main()
#     print(f"Result: ")

# if __name__ == "__main__":
#     demonstrate()
# '''

from typing import Dict, Any, List, Optional
import openai
import os
from dotenv import load_dotenv
import sys
from io import StringIO
import re
import traceback
import ast
import importlib.util
import subprocess


def remove_backticks(text: str) -> str:
    """Remove code block backticks from text"""
    # Handle multiple code blocks and different languages
    patterns = [
        r"```python\n?(.*?)\n?```",
        r"```\n?(.*?)\n?```",
        r"`(.*?)`"
    ]
    
    for pattern in patterns:
        if re.search(pattern, text, flags=re.DOTALL):
            return re.sub(pattern, r"\1", text, flags=re.DOTALL).strip()
    
    return text.strip()


def extract_imports(code: str) -> List[str]:
    """Extract import statements from code"""
    try:
        tree = ast.parse(code)
        imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)
        
        return imports
    except:
        # Fallback regex method
        import_pattern = r'(?:from\s+(\w+)|import\s+(\w+))'
        matches = re.findall(import_pattern, code)
        imports = []
        for match in matches:
            imports.extend([m for m in match if m])
        return imports


def check_module_availability(modules: List[str]) -> Dict[str, bool]:
    """Check which modules are available"""
    available = {}
    for module in modules:
        try:
            importlib.util.find_spec(module)
            available[module] = True
        except (ImportError, ModuleNotFoundError, ValueError):
            available[module] = False
    
    return available


# Load environment variables
load_dotenv()


class CodeAgent:
    def __init__(self):
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.common_modules = [
            'numpy', 'pandas', 'matplotlib', 'seaborn', 'sklearn', 
            'requests', 'json', 'csv', 'sqlite3', 'datetime', 
            'random', 'math', 'os', 're', 'collections', 'itertools',
            'functools', 'operator', 'statistics', 'pathlib'
        ]
        self.available_modules = check_module_availability(self.common_modules)
    
    def execute(self, task: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate and execute code based on the task"""
        
        try:
            # Generate code with validation
            code = self._generate_code(task, context)
            
            # Pre-validate code for common issues
            validation_result = self._validate_code(code)
            
            if not validation_result['valid']:
                # Try to fix common issues and regenerate
                code = self._fix_and_regenerate_code(task, context, validation_result['issues'])
            
            # Test code execution
            execution_result = self._test_code(code)
            
            # Generate documentation
            documentation = self._generate_documentation(code, task)
            
            return {
                "code": code,
                "execution_result": execution_result,
                "documentation": documentation,
                "language": "python",
                "status": "completed",
                "validation": validation_result
            }
        
        except Exception as e:
            return {
                "error": str(e),
                "traceback": traceback.format_exc(),
                "status": "failed",
                "fallback_code": self._generate_simple_fallback(task)
            }
    
    def _generate_code(self, task: str, context: Dict[str, Any] = None) -> str:
        """Generate code implementation using OpenAI API"""
        
        context_info = ""
        if context and context.get("research"):
            research_data = context["research"]
            context_info = f"""
            Research Context:
            {research_data.get('synthesis', '')[:1000]}
            """
        
        # Create available modules list
        available_libs = [lib for lib, available in self.available_modules.items() if available]
        
        prompt = f"""
        Generate Python code for this task:
        Task: {task}
        
        {context_info}
        
        Available libraries: {', '.join(available_libs)}
        
        Requirements:
        - Write clean, well-documented Python code
        - Use only standard libraries or the available libraries listed above
        - Include proper error handling and input validation
        - Make the code executable with clear demonstration
        - Add comments explaining key logic
        - Define all functions and variables properly
        - Include a main execution block with if __name__ == "__main__":
        - Handle edge cases appropriately
        - Use descriptive variable names
        
        Code Structure:
        - Import statements at the top
        - Function definitions
        - Main logic
        - Demonstration/example usage
        
        Output format:
        ```python
        # Your code here
        ```
        
        IMPORTANT: 
        - Return ONLY executable Python code in backticks
        - Do NOT include explanations outside the code
        - Ensure all variables are defined before use
        - Test that imports are available
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",  # Using more reliable model
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000,
                temperature=0.1  # Lower temperature for more consistent code
            )
            
            return remove_backticks(response.choices[0].message.content)
        
        except Exception as e:
            print(f"Code generation error: {e}")
            return self._generate_simple_fallback(task)
    
    def _validate_code(self, code: str) -> Dict[str, Any]:
        """Validate code for common issues"""
        issues = []
        
        try:
            # Parse AST to check syntax
            ast.parse(code)
        except SyntaxError as e:
            issues.append(f"Syntax error: {str(e)}")
            return {"valid": False, "issues": issues}
        
        # Check for undefined variables (basic check)
        try:
            # Extract imports
            imports = extract_imports(code)
            
            # Check if imports are available
            unavailable_imports = []
            for imp in imports:
                if imp in self.available_modules and not self.available_modules[imp]:
                    unavailable_imports.append(imp)
            
            if unavailable_imports:
                issues.append(f"Unavailable imports: {', '.join(unavailable_imports)}")
        
        except Exception as e:
            issues.append(f"Import validation error: {str(e)}")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues
        }
    
    def _fix_and_regenerate_code(self, task: str, context: Dict[str, Any], issues: List[str]) -> str:
        """Fix common issues and regenerate code"""
        
        issue_description = "\n".join(f"- {issue}" for issue in issues)
        available_libs = [lib for lib, available in self.available_modules.items() if available]
        
        prompt = f"""
        The previous code had these issues:
        {issue_description}
        
        Task: {task}
        Available libraries ONLY: {', '.join(available_libs)}
        
        Generate corrected Python code that:
        - Fixes all the mentioned issues
        - Uses ONLY the available libraries listed above
        - Includes proper error handling
        - Has all variables properly defined
        - Is syntactically correct and executable
        
        ```python
        # Corrected code here
        ```
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000,
                temperature=0.1
            )
            
            return remove_backticks(response.choices[0].message.content)
        
        except Exception:
            return self._generate_simple_fallback(task)
    
    def _test_code(self, code: str) -> Dict[str, Any]:
        """Safely test the generated code"""
        
        try:
            # Capture output
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            sys.stdout = captured_output = StringIO()
            sys.stderr = captured_error = StringIO()
            
            # Create safe execution environment
            safe_builtins = {
                'print': print, 'len': len, 'range': range, 'str': str, 'int': int,
                'float': float, 'list': list, 'dict': dict, 'tuple': tuple, 'set': set,
                'enumerate': enumerate, 'sum': sum, 'zip': zip, 'min': min, 'max': max,
                'abs': abs, 'round': round, 'sorted': sorted, 'reversed': reversed,
                'map': map, 'filter': filter, 'all': all, 'any': any, 'type': type,
                'isinstance': isinstance, 'hasattr': hasattr, 'getattr': getattr,
                'setattr': setattr, 'chr': chr, 'ord': ord, 'bin': bin, 'hex': hex,
                'oct': oct, 'pow': pow, 'divmod': divmod
            }
            
            exec_globals = {'__builtins__': safe_builtins}
            
            # Safely import commonly used modules
            safe_modules = {
                'math': 'math', 'random': 'random', 'datetime': 'datetime',
                'json': 'json', 'os': 'os', 're': 're', 'sys': 'sys',
                'collections': 'collections', 'itertools': 'itertools',
                'functools': 'functools', 'operator': 'operator',
                'statistics': 'statistics', 'pathlib': 'pathlib'
            }
            
            for name, module_name in safe_modules.items():
                try:
                    exec_globals[name] = __import__(module_name)
                except ImportError:
                    pass
            
            # Try to import numpy and other common data science libraries
            try:
                import numpy as np
                exec_globals['np'] = np
                exec_globals['numpy'] = np
            except ImportError:
                pass
            
            try:
                import pandas as pd
                exec_globals['pd'] = pd
                exec_globals['pandas'] = pd
            except ImportError:
                pass
            
            # Execute the code
            exec(code, exec_globals)
            
            # Restore stdout/stderr
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            
            output = captured_output.getvalue()
            error_output = captured_error.getvalue()
            
            return {
                "success": True,
                "output": output.strip() if output.strip() else "Code executed successfully (no output)",
                "error": error_output.strip() if error_output.strip() else None
            }
        
        except Exception as e:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            
            return {
                "success": False,
                "output": None,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    def _generate_documentation(self, code: str, task: str) -> str:
        """Generate documentation for the code"""
        
        prompt = f"""
        Generate concise documentation for this code:
        
        Task: {task}
        Code:
        ```python
        {code}
        ```
        
        Provide:
        1. Brief description (1-2 sentences)
        2. Key functions/classes and their purpose
        3. Usage example or how to run
        4. Input/output description
        5. Any important notes
        
        Keep it concise and practical.
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=600,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
        
        except Exception:
            return f"Code implementation for: {task}\n\nThis code provides a solution to the requested task with proper error handling and demonstration."
    
    def _generate_simple_fallback(self, task: str) -> str:
        """Generate a simple fallback implementation"""
        
        task_lower = task.lower()
        
        # Basic template that works for most tasks
        return f'''
def solve_task():
    """
    Implementation for: {task}
    """
    print("Task: {task}")
    
    # Basic implementation structure
    try:
        # Main logic would go here
        result = "Task processing completed"
        
        print(f"Processing: {task}")
        print("Implementation ready for enhancement")
        
        return result
        
    except Exception as e:
        print(f"Error: {{e}}")
        return None

def main():
    """Main execution function"""
    print("=" * 50)
    print("Code Agent - Task Implementation")
    print("=" * 50)
    
    result = solve_task()
    
    if result:
        print(f"\\nResult: {{result}}")
        print("\\nStatus: Ready for further development")
    else:
        print("\\nTask needs additional specification")
    
    print("=" * 50)

if __name__ == "__main__":
    main()
'''


