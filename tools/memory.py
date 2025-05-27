import json
import os
from typing import Dict, Any, Optional
from datetime import datetime

class MemoryStore:
    def __init__(self, storage_file: str = "memory_store.json"):
        self.storage_file = storage_file
        self.memory = self._load_memory()
    
    def _load_memory(self) -> Dict[str, Any]:
        """Load memory from file"""
        if os.path.exists(self.storage_file):
            try:
                with open(self.storage_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading memory: {e}")
                return {}
        return {}
    
    def _save_memory(self):
        """Save memory to file"""
        try:
            with open(self.storage_file, 'w') as f:
                json.dump(self.memory, f, indent=2, default=str)
        except Exception as e:
            print(f"Error saving memory: {e}")
    
    def store_task(self, task_id: str, task_data: Dict[str, Any]):
        """Store task data"""
        task_data['last_updated'] = datetime.now().isoformat()
        self.memory[task_id] = task_data
        self._save_memory()
    
    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve task data"""
        return self.memory.get(task_id)
    
    def list_tasks(self) -> Dict[str, Any]:
        """List all stored tasks"""
        return {
            task_id: {
                'task': data.get('task', ''),
                'status': data.get('status', ''),
                'last_updated': data.get('last_updated', '')
            }
            for task_id, data in self.memory.items()
        }
    
    def clear_memory(self):
        """Clear all stored memory"""
        self.memory = {}
        self._save_memory()
    
    def store_context(self, key: str, context: Dict[str, Any]):
        """Store contextual information"""
        if 'contexts' not in self.memory:
            self.memory['contexts'] = {}
        
        self.memory['contexts'][key] = {
            'data': context,
            'timestamp': datetime.now().isoformat()
        }
        self._save_memory()
    
    def get_context(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieve contextual information"""
        contexts = self.memory.get('contexts', {})
        context_data = contexts.get(key)
        return context_data['data'] if context_data else None