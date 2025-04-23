from typing import Dict, List, Any, Optional
import json
import os
from datetime import datetime
from langchain.llms import HuggingFacePipeline
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model
import torch
from PIL import Image
import numpy as np

class ChatAgent:
    """
    Agent responsible for handling user interactions and managing the analysis pipeline.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the ChatAgent."""
        self.config = config or {}
        self.conversation_history = []
        self.current_session = None
        
        # Initialize Phi-2
        phi2_tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
        phi2_model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2")
        phi2_pipe = pipeline(
            "text-generation",
            model=phi2_model,
            tokenizer=phi2_tokenizer,
            max_length=500
        )
        self.phi2_llm = HuggingFacePipeline(pipeline=phi2_pipe)
        
        # Initialize Mistral
        mistral_tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
        mistral_model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
        mistral_pipe = pipeline(
            "text-generation",
            model=mistral_model,
            tokenizer=mistral_tokenizer,
            max_length=1000
        )
        self.mistral_llm = HuggingFacePipeline(pipeline=mistral_pipe)
        
        # Initialize LLaVA
        model_path = "liuhaotian/llava-v1.5-13b"
        model_name = get_model_name_from_path(model_path)
        self.llava_tokenizer, self.llava_model, self.llava_image_processor, self.llava_context_len = load_pretrained_model(
            model_path, None, model_name
        )
        
        # Create chains
        self.analysis_chain = LLMChain(
            llm=self.phi2_llm,
            prompt=PromptTemplate(
                input_variables=["question", "context"],
                template="""
                Based on the following context, answer the question:
                
                Context: {context}
                
                Question: {question}
                
                Please provide a detailed and accurate answer.
                """
            )
        )
        
        self.explanation_chain = LLMChain(
            llm=self.mistral_llm,
            prompt=PromptTemplate(
                input_variables=["question", "context"],
                template="""
                Explain the following in detail:
                
                Context: {context}
                
                Question: {question}
                
                Please provide a comprehensive explanation with examples and relevant details.
                """
            )
        )
        
    def start_session(self, session_id: str = None) -> Dict[str, Any]:
        """Start a new chat session."""
        if session_id is None:
            session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
        self.current_session = {
            'id': session_id,
            'start_time': datetime.now().isoformat(),
            'status': 'active',
            'messages': []
        }
        
        return self.current_session
        
    def process_command(self, command: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process a user command."""
        if not self.current_session:
            return {'error': 'No active session. Call start_session() first.'}
            
        params = params or {}
        
        try:
            if command == 'load_file':
                return self._handle_load_file(params)
            elif command == 'analyze':
                return self._handle_analyze(params)
            elif command == 'get_results':
                return self._handle_get_results(params)
            elif command == 'help':
                return self._handle_help()
            else:
                return {'error': f'Unknown command: {command}'}
                
        except Exception as e:
            return {'error': str(e)}
            
    def _handle_load_file(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle file loading command."""
        if 'vsi_path' not in params or 'xml_path' not in params:
            return {'error': 'Missing required parameters: vsi_path and xml_path'}
            
        # Add to conversation history
        self.current_session['messages'].append({
            'type': 'command',
            'command': 'load_file',
            'params': params,
            'timestamp': datetime.now().isoformat()
        })
        
        return {'status': 'success', 'message': 'Files loaded successfully'}
        
    def _handle_analyze(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle analysis command."""
        if 'tile_size' not in params or 'confidence' not in params:
            return {'error': 'Missing required parameters: tile_size and confidence'}
            
        # Add to conversation history
        self.current_session['messages'].append({
            'type': 'command',
            'command': 'analyze',
            'params': params,
            'timestamp': datetime.now().isoformat()
        })
        
        return {'status': 'success', 'message': 'Analysis started'}
        
    def _handle_get_results(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle results retrieval command."""
        # Add to conversation history
        self.current_session['messages'].append({
            'type': 'command',
            'command': 'get_results',
            'params': params,
            'timestamp': datetime.now().isoformat()
        })
        
        return {'status': 'success', 'message': 'Results retrieved'}
        
    def _handle_help(self) -> Dict[str, Any]:
        """Handle help command."""
        help_text = """
        Available commands:
        - load_file: Load VSI and XML files
          Parameters: vsi_path, xml_path
          
        - analyze: Start analysis
          Parameters: tile_size, confidence
          
        - get_results: Get analysis results
          Parameters: None
          
        - help: Show this help message
          Parameters: None
        """
        
        return {'status': 'success', 'message': help_text}
        
    def answer_question(self, question: str, context: Dict[str, Any] = None, 
                       image: Optional[np.ndarray] = None) -> str:
        """Answer a user question using the appropriate model."""
        context = context or {}
        
        if image is not None:
            # Use LLaVA for image-related questions
            image_pil = Image.fromarray(image)
            prompt = f"""
            Based on this image and the following context, answer the question:
            
            Context: {json.dumps(context, indent=2)}
            
            Question: {question}
            
            Please provide a detailed answer that incorporates both the image content and the context.
            """
            
            response = eval_model(
                self.llava_model,
                self.llava_tokenizer,
                self.llava_image_processor,
                prompt,
                image_pil,
                self.llava_context_len
            )
        else:
            # Use appropriate chain based on question type
            if "explain" in question.lower() or "how" in question.lower():
                response = self.explanation_chain.run(
                    question=question,
                    context=json.dumps(context, indent=2)
                )
            else:
                response = self.analysis_chain.run(
                    question=question,
                    context=json.dumps(context, indent=2)
                )
                
        # Add to conversation history
        if self.current_session:
            self.current_session['messages'].append({
                'type': 'question',
                'question': question,
                'response': response,
                'timestamp': datetime.now().isoformat()
            })
            
        return response
        
    def save_session(self, output_dir: str) -> str:
        """Save the current session to a file."""
        if not self.current_session:
            raise ValueError("No active session to save")
            
        os.makedirs(output_dir, exist_ok=True)
        session_file = os.path.join(output_dir, f"{self.current_session['id']}.json")
        
        with open(session_file, 'w') as f:
            json.dump(self.current_session, f, indent=2)
            
        return session_file
        
    def load_session(self, session_file: str) -> Dict[str, Any]:
        """Load a previous session from a file."""
        with open(session_file, 'r') as f:
            session = json.load(f)
            
        self.current_session = session
        return session 