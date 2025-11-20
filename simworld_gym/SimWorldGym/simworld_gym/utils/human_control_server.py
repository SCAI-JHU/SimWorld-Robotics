"""
Human Control Panel: Simple tkinter GUI for HumanInterface.

Features:
- Simple GUI with buttons for movement controls (Forward/Back/Left/Right, Turn Left/Right)
- Real-time status display showing cooldown and last action
- Chat interface with OpenAI API for conversational control
- Keyboard shortcuts (WASD, QE)

Usage:
    from simworld_gym.utils.unrealcv_basic import UnrealCV
    from simworld_gym.utils.human_interface import HumanInterface
    from simworld_gym.utils.human_control_server import HumanControlPanel

    ucv = UnrealCV(port=9900, ip="127.0.0.1", resolution=(320, 240))
    human = HumanInterface(ucv, actor_name="Human01")
    panel = HumanControlPanel(human)
    panel.run()
"""
import json
import os
import threading
import tkinter as tk
from tkinter import scrolledtext, ttk
from datetime import datetime
from typing import Optional

from openai import OpenAI

from .human_interface import HumanInterface


class HumanControlPanel:
    """Simple tkinter GUI for controlling HumanInterface with buttons and AI chat."""

    def __init__(self, human_interface: HumanInterface, openai_api_key: Optional[str] = None, 
                 openai_model: str = "gpt-4o-mini"):
        self.human = human_interface
        
        # Setup OpenAI client
        api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        if api_key:
            self.openai_client = OpenAI(api_key=api_key)
            self.openai_model = openai_model
        else:
            self.openai_client = None
            print("⚠️  Warning: OpenAI API key not provided. Chat feature will be disabled.")
        
        # Create main window
        self.root = tk.Tk()
        self.root.title(f"🎮 Human Control Panel - {human_interface.actor_name}")
        self.root.geometry("600x700")
        self.root.resizable(False, False)
        
        self._setup_ui()
        self._bind_keys()
        self._start_status_update()

    def _setup_ui(self):
        """Setup the GUI layout."""
        # Status Frame
        status_frame = ttk.LabelFrame(self.root, text="📊 Status", padding=10)
        status_frame.pack(fill="x", padx=10, pady=5)
        
        status_grid = ttk.Frame(status_frame)
        status_grid.pack(fill="x")
        
        ttk.Label(status_grid, text="Actor:").grid(row=0, column=0, sticky="w", padx=5)
        self.actor_label = ttk.Label(status_grid, text=self.human.actor_name, font=("Arial", 10, "bold"))
        self.actor_label.grid(row=0, column=1, sticky="w", padx=5)
        
        ttk.Label(status_grid, text="State:").grid(row=0, column=2, sticky="w", padx=5)
        self.state_label = ttk.Label(status_grid, text="Ready", foreground="green", font=("Arial", 10, "bold"))
        self.state_label.grid(row=0, column=3, sticky="w", padx=5)
        
        ttk.Label(status_grid, text="Last Action:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.action_label = ttk.Label(status_grid, text="None", font=("Arial", 10))
        self.action_label.grid(row=1, column=1, columnspan=3, sticky="w", padx=5, pady=5)
        
        # Control Frame
        control_frame = ttk.LabelFrame(self.root, text="🕹️ Movement Controls", padding=10)
        control_frame.pack(fill="x", padx=10, pady=5)
        
        # Keyboard hint
        hint_label = ttk.Label(control_frame, 
                              text="💡 Keyboard: W/S=Forward/Back | A/D=Left/Right | Q/E=Turn Left/Right",
                              foreground="blue", font=("Arial", 9))
        hint_label.pack(pady=(0, 10))
        
        # Button grid
        btn_frame = ttk.Frame(control_frame)
        btn_frame.pack()
        
        # Create buttons with grid layout
        btn_style = {"width": 12, "padding": 5}
        
        # Row 0: Forward
        self.btn_forward = ttk.Button(btn_frame, text="⬆️ Forward (W)", 
                                      command=lambda: self._execute_action("forward"), **btn_style)
        self.btn_forward.grid(row=0, column=1, padx=5, pady=5)
        
        # Row 1: Left, Backward, Right
        self.btn_left = ttk.Button(btn_frame, text="⬅️ Left (A)", 
                                   command=lambda: self._execute_action("left"), **btn_style)
        self.btn_left.grid(row=1, column=0, padx=5, pady=5)
        
        self.btn_backward = ttk.Button(btn_frame, text="⬇️ Back (S)", 
                                       command=lambda: self._execute_action("backward"), **btn_style)
        self.btn_backward.grid(row=1, column=1, padx=5, pady=5)
        
        self.btn_right = ttk.Button(btn_frame, text="➡️ Right (D)", 
                                    command=lambda: self._execute_action("right"), **btn_style)
        self.btn_right.grid(row=1, column=2, padx=5, pady=5)
        
        # Row 2: Turn Left, Turn Right
        self.btn_turn_left = ttk.Button(btn_frame, text="↩️ Turn Left (Q)", 
                                        command=lambda: self._execute_action("turn_left"), **btn_style)
        self.btn_turn_left.grid(row=2, column=0, padx=5, pady=5)
        
        self.btn_turn_right = ttk.Button(btn_frame, text="↪️ Turn Right (E)", 
                                         command=lambda: self._execute_action("turn_right"), **btn_style)
        self.btn_turn_right.grid(row=2, column=2, padx=5, pady=5)
        
        # Store buttons for easy access
        self.control_buttons = [
            self.btn_forward, self.btn_backward, self.btn_left, 
            self.btn_right, self.btn_turn_left, self.btn_turn_right
        ]
        
        # Chat Frame
        chat_frame = ttk.LabelFrame(self.root, text="💬 AI Assistant Chat", padding=10)
        chat_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Chat messages area
        self.chat_text = scrolledtext.ScrolledText(chat_frame, wrap=tk.WORD, height=15, 
                                                    font=("Arial", 10), state="disabled")
        self.chat_text.pack(fill="both", expand=True, pady=(0, 10))
        
        # Configure tags for styling
        self.chat_text.tag_config("user", foreground="blue", font=("Arial", 10, "bold"))
        self.chat_text.tag_config("assistant", foreground="green", font=("Arial", 10, "bold"))
        self.chat_text.tag_config("system", foreground="gray", font=("Arial", 9, "italic"))
        self.chat_text.tag_config("time", foreground="gray", font=("Arial", 8))
        
        # Add welcome message
        self._add_chat_message("assistant", 
            "Hi! I'm your AI assistant. You can chat with me or ask me to control the character.\n"
            "Try saying 'move forward' or 'turn left'!")
        
        # Chat input area
        input_frame = ttk.Frame(chat_frame)
        input_frame.pack(fill="x")
        
        self.chat_input = ttk.Entry(input_frame, font=("Arial", 10))
        self.chat_input.pack(side="left", fill="x", expand=True, padx=(0, 5))
        self.chat_input.bind("<Return>", lambda e: self._send_chat_message())
        
        self.btn_send = ttk.Button(input_frame, text="Send", command=self._send_chat_message, width=10)
        self.btn_send.pack(side="right")
        
        if not self.openai_client:
            self.chat_input.config(state="disabled")
            self.btn_send.config(state="disabled")
            self._add_chat_message("system", "OpenAI API not configured. Chat is disabled.")

    def _bind_keys(self):
        """Bind keyboard shortcuts."""
        key_map = {
            'w': lambda: self._execute_action("forward"),
            's': lambda: self._execute_action("backward"),
            'a': lambda: self._execute_action("left"),
            'd': lambda: self._execute_action("right"),
            'q': lambda: self._execute_action("turn_left"),
            'e': lambda: self._execute_action("turn_right"),
        }
        
        for key, action in key_map.items():
            self.root.bind(f'<{key}>', lambda e, a=action: a())
            self.root.bind(f'<{key.upper()}>', lambda e, a=action: a())

    def _execute_action(self, action_name: str):
        """Execute a movement action."""
        if self.human.is_busy():
            return
        
        action_map = {
            'forward': self.human.move_forward,
            'backward': self.human.move_backward,
            'left': self.human.move_left,
            'right': self.human.move_right,
            'turn_left': self.human.turn_left,
            'turn_right': self.human.turn_right,
        }
        
        if action_name in action_map:
            success = action_map[action_name]()
            if success:
                self.action_label.config(text=action_name.replace('_', ' ').title())
                self._update_button_states()

    def _update_button_states(self):
        """Update button states based on busy status."""
        is_busy = self.human.is_busy()
        state = "disabled" if is_busy else "normal"
        
        for btn in self.control_buttons:
            btn.config(state=state)
        
        if is_busy:
            self.state_label.config(text="Busy", foreground="red")
        else:
            self.state_label.config(text="Ready", foreground="green")

    def _start_status_update(self):
        """Start periodic status updates."""
        def update():
            self._update_button_states()
            self.root.after(100, update)
        
        update()

    def _add_chat_message(self, role: str, content: str):
        """Add a message to the chat display."""
        self.chat_text.config(state="normal")
        
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        if role == "user":
            self.chat_text.insert(tk.END, f"[{timestamp}] You:\n", "time")
            self.chat_text.insert(tk.END, f"{content}\n\n", "user")
        elif role == "assistant":
            self.chat_text.insert(tk.END, f"[{timestamp}] AI:\n", "time")
            self.chat_text.insert(tk.END, f"{content}\n\n", "assistant")
        else:  # system
            self.chat_text.insert(tk.END, f"[{timestamp}] System:\n", "time")
            self.chat_text.insert(tk.END, f"{content}\n\n", "system")
        
        self.chat_text.see(tk.END)
        self.chat_text.config(state="disabled")

    def _send_chat_message(self):
        """Send a chat message to OpenAI."""
        if not self.openai_client:
            return
        
        message = self.chat_input.get().strip()
        if not message:
            return
        
        self.chat_input.delete(0, tk.END)
        self._add_chat_message("user", message)
        
        # Disable input while processing
        self.chat_input.config(state="disabled")
        self.btn_send.config(state="disabled")
        self._add_chat_message("system", "AI is thinking...")
        
        # Process in background thread
        def process():
            try:
                response = self.openai_client.chat.completions.create(
                    model=self.openai_model,
                    messages=[
                        {
                            "role": "system",
                            "content": """You are a helpful AI assistant controlling a character in a simulator. 
You can chat naturally and also execute movement commands when asked.

Available actions:
- move_forward: Move the character forward
- move_backward: Move the character backward  
- move_left: Move the character to the left
- move_right: Move the character to the right
- turn_left: Rotate the character 90° left
- turn_right: Rotate the character 90° right

When the user asks you to move or control the character, use the appropriate function.
Be conversational and helpful!"""
                        },
                        {"role": "user", "content": message}
                    ],
                    functions=[
                        {
                            "name": "execute_action",
                            "description": "Execute a movement or rotation action for the character",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "action": {
                                        "type": "string",
                                        "enum": ["forward", "backward", "left", "right", "turn_left", "turn_right"],
                                        "description": "The action to execute"
                                    }
                                },
                                "required": ["action"]
                            }
                        }
                    ],
                    function_call="auto",
                    temperature=0.7,
                    max_tokens=256
                )
                
                msg = response.choices[0].message
                action_executed = None
                
                # Check if function was called
                if msg.function_call:
                    function_args = json.loads(msg.function_call.arguments)
                    action_name = function_args.get('action')
                    
                    action_map = {
                        'forward': self.human.move_forward,
                        'backward': self.human.move_backward,
                        'left': self.human.move_left,
                        'right': self.human.move_right,
                        'turn_left': self.human.turn_left,
                        'turn_right': self.human.turn_right,
                    }
                    
                    if action_name in action_map:
                        success = action_map[action_name]()
                        if success:
                            action_executed = action_name
                            response_text = f"Sure! I've executed: {action_name.replace('_', ' ')}."
                            self.root.after(0, lambda: self.action_label.config(
                                text=action_name.replace('_', ' ').title()))
                        else:
                            response_text = "The character is currently busy. Please wait a moment and try again."
                    else:
                        response_text = f"I don't recognize that action: {action_name}"
                else:
                    response_text = msg.content or "I'm not sure how to respond to that."
                
                # Update UI in main thread
                self.root.after(0, lambda: self._handle_chat_response(response_text))
                
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                self.root.after(0, lambda: self._handle_chat_response(error_msg))
        
        thread = threading.Thread(target=process, daemon=True)
        thread.start()

    def _handle_chat_response(self, response: str):
        """Handle chat response in main thread."""
        # Remove "thinking" message
        self.chat_text.config(state="normal")
        content = self.chat_text.get("1.0", tk.END)
        if "AI is thinking..." in content:
            lines = content.split("\n")
            # Remove last 3 lines (timestamp, system message, empty line)
            new_content = "\n".join(lines[:-3]) + "\n"
            self.chat_text.delete("1.0", tk.END)
            self.chat_text.insert("1.0", new_content)
        self.chat_text.config(state="disabled")
        
        # Add response
        self._add_chat_message("assistant", response)
        
        # Re-enable input
        self.chat_input.config(state="normal")
        self.btn_send.config(state="normal")
        self.chat_input.focus()

    def run(self):
        """Start the GUI main loop."""
        print(f"🚀 Starting Human Control Panel")
        print(f"   Actor: {self.human.actor_name}")
        print(f"   OpenAI: {'Enabled ✓' if self.openai_client else 'Disabled ✗'}")
        print("\n💡 Use WASD and QE keys or click buttons to control the character!")
        
        self.root.mainloop()


# Convenience function to quickly start panel
def start_control_panel(unrealcv_instance, actor_name: str, openai_api_key: Optional[str] = None):
    """Quick start function to create and run the control panel."""
    from .human_interface import HumanInterface
    
    human = HumanInterface(unrealcv_instance, actor_name)
    panel = HumanControlPanel(human, openai_api_key=openai_api_key)
    panel.run()

