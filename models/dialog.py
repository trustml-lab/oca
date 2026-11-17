import os, sys
import numpy as np
from tqdm import tqdm

from getpass import getpass
import openai

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.adapters.openai import convert_message_to_dict

from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import add_messages
from typing import Annotated
from typing_extensions import TypedDict
import inspect

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

from models.scg import OSGLearner, OSGAdvBanditLearner, OSGSemiAdvBanditLearner
import json
import pickle


CHECKPOINT_FILENAME = 'checkpoint.pkl'


def _get_openai_api_key():
    """Return OPENAI_API_KEY, prompting only when an OpenAI client is actually created.

    The bandit experiments (main.py) import this module but never call the OpenAI API,
    so the key must not be requested at import time.
    """
    if not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = getpass("Please provide your OPENAI_API_KEY: ")
    return os.environ["OPENAI_API_KEY"]
    
    
class DialogLearner:
    system_prompt = inspect.cleandoc(
        """\
        You are a helpful agent who continuously asks questions related to the given context. The following is the source context for question answering:
        
        Context:
        {context}

        {instructions}

        When you are done with the current context, say 'SHIFT' to request a new context. 
        You must decide this **yourself** — do **not** ask whether to 'SHIFT', simply respond with 'SHIFT' by yourself when needed.
        """
        # When you are finished with the conversation, respond with a single word 'FINISHED'.
        # However, do **not** respond with 'FINISHED' unless a **serious problem** occurs that prevents you from continuing the conversation.
        # """
    )
    instructions = inspect.cleandoc(
        """\
        You are an agent trying to question about contexts to the user. The question should be answerable by the context.
        Keep generating new, relevant questions based on the context.
        
        You may also refer back to the user's previous answers (e.g., ask for clarifications or follow-ups), but you must always end your message with a question **related to the context.
        Do **not** end your message with general or casual questions (e.g., greetings, personal inquiries or asking shift). 
        Your question must always be **strictly related to the context**.

        You must always **ask**, not **answer**.
        
        The user cannot see the contexts. You must not reveal any part of the context to the user, nor suggest that they ask about it.
        """
    )
    
    def __init__(self, model, params=None):
        self.labeler = Labeler(model, params)
        self.params = params
        self.T = params.T
        self.x_split = []
        self.shift = []
        self.max_turn = 20
        self.client = openai.OpenAI(api_key=_get_openai_api_key())
        
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
        
        graph_builder = StateGraph(State)
        graph_builder.add_node("user", self.simulated_user_node)
        graph_builder.add_node("chat_bot", self.chat_bot_node)
        graph_builder.add_node("labeler_osg", self.labeler)
        # Every response from  your chat bot will automatically go to the
        # simulated user
        graph_builder.add_edge("chat_bot", "labeler_osg")
        # graph_builder.add_edge("labeler_osg", "user")
        graph_builder.add_conditional_edges(
            "user",
            self.should_continue,
            {
                "shift": "user",
                "continue": "chat_bot",
            },
        )
        graph_builder.add_conditional_edges(
            "labeler_osg",
            self.should_continue_labeler,
            {
                "end": END,
                "continue": "user",
            },
        )
        
        graph_builder.add_edge(START, "user")
        self.simulation = graph_builder.compile()
        self._dbg_enabled = bool(os.environ.get("OSG_DEBUG")) or bool(getattr(params, "dbg", False))
        self._dbg_printed = set()
        
        
    def simulate(self, rd, updated_params=None):
        # self.params = updated_params if updated_params else self.params

        self.iterd = iter(rd)
        self._iterd_idx = 0
        self._chunk_log = []

        ckpt = self._load_checkpoint()
        init_state = {"messages": []}
        if ckpt is not None:
            self.labeler.osg = ckpt['osg']
            self.labeler.x_idx = ckpt['labeler_x_idx']
            self.x_split = ckpt['x_split']
            self.shift = ckpt['shift']
            self.id = ckpt['id']
            self.title = ckpt['title']
            self.context = ckpt['context']
            self._chunk_log = ckpt['chunk_log']
            for _ in range(ckpt['iterd_idx']):
                next(self.iterd)
            self._iterd_idx = ckpt['iterd_idx']
            if ckpt.get('last_state'):
                init_state = ckpt['last_state']
            print(f"[checkpoint] resumed: x_idx={self.labeler.x_idx}, iterd_idx={self._iterd_idx}")
        else:
            init_data = next(self.iterd)
            self._iterd_idx = 1
            self.id = init_data["id"]
            self.title = init_data["title"]
            self.context = init_data["context"]

        chunk_log = self._chunk_log
        pbar = tqdm(total=self.T, initial=self.labeler.x_idx, desc="Executing states")

        last_full_state = init_state
        pending_save = False
        try:
            for stream_item in self.simulation.stream(init_state, config={"recursion_limit": 50000}, stream_mode=["updates", "values"]):
                mode, chunk = stream_item
                if mode == "values":
                    last_full_state = chunk
                    if pending_save:
                        self._save_checkpoint(last_full_state)
                        pending_save = False
                    continue

                # mode == "updates"
                if END not in chunk:
                    tqdm.write(str(chunk))
                    tqdm.write("---")


                    chunk_log.append(chunk)

                if "labeler_osg" in chunk:
                    tqdm.write("="*30)
                    pbar.update(1)
                    pending_save = True
                    
                    fdata = 0
                    for data in chunk_log:
                        if 'labeler_osg' in data and data['labeler_osg']['label'] == 0:
                            fdata += 1
                    # If the labeler failed early, avoid crashing the run during plotting.
                    ylim = (fdata / self.labeler.x_idx) if self.labeler.x_idx > 0 else self.params.alpha
                            
                    try:
                        with PdfPages(os.path.join(self.params.output_dir, 'fdrs_plot.pdf')) as pdf:
                            plt.figure()
                            if isinstance(self.labeler.osg, OSGSemiAdvBanditLearner):
                                plt.plot(self.labeler.osg.fdrs, label=r'$\mathtt{ExAUL}$', color='green')
                            elif isinstance(self.labeler.osg, OSGAdvBanditLearner):
                                plt.plot(self.labeler.osg.fdrs, label=r'$\mathtt{Exp3\text{-}CA}$', color='red')
                            elif isinstance(self.labeler.osg, OSGLearner):
                                plt.plot(self.labeler.osg.fdrs, label=r'$\mathtt{EW\text{-}CA}$', color='blue')
                            
                            plt.plot(self.labeler.osg.emp_fdrs, label=r'$\mathtt{No\text{-}CA}$', color='orange')
                            plt.xlabel(r'$t$',fontsize=15)
                            plt.ylabel('FDR',fontsize=15)
                            if abs(ylim - self.params.alpha) < 1e-12:
                                ylo = max(0.0, self.params.alpha - 0.05)
                                yhi = min(1.0, self.params.alpha + 0.05)
                            else:
                                ylo = self.params.alpha - (ylim - self.params.alpha) * 0.9
                                yhi = ylim + (ylim - self.params.alpha) * 0.9
                            plt.ylim(ylo, yhi)
                            # plt.title('FDR')
                            # plt.axhline(y=ylim, color='k', alpha=0.7, linestyle='dotted', label='Empirical FDR upper-bound')
                            plt.axhline(y=self.params.alpha, color='k', linestyle='dashed', label=r'$\alpha$='+f'{self.params.alpha}')

                            plt.legend(fontsize=10, loc='upper right')
                            plt.grid(True, which='both', linestyle='--', linewidth=0.5, zorder=1)
                            plt.tight_layout()
                            
                            # plt.axvspan(plt.xlim()[0], x_split_points[0], color='red', alpha=0.3)
                            # for i in x_split_points:
                            #     plt.axvline(x=i, color='red', linestyle='dotted', alpha=0.7, zorder=1)
                                # if i % 3 == 0: color = 'red'
                                # elif i % 3 == 1: color = 'blue'
                                # else: color = 'green'
                            plt.savefig(os.path.join(self.params.output_dir, 'fdrs_plot.png'))
                            pdf.savefig(bbox_inches='tight')
                            plt.close()
                        
                        with PdfPages(os.path.join(self.params.output_dir, 'inefficiencies_plot.pdf')) as pdf:
                            plt.figure()
                            if isinstance(self.labeler.osg, OSGSemiAdvBanditLearner):
                                plt.plot(1-np.array(self.labeler.osg.efficiencies), label=r'$\mathtt{ExAUL}$', color='green')
                            elif isinstance(self.labeler.osg, OSGAdvBanditLearner):
                                plt.plot(1-np.array(self.labeler.osg.efficiencies), label=r'$\mathtt{Exp3\text{-}CA}$', color='red')
                            elif isinstance(self.labeler.osg, OSGLearner):
                                plt.plot(1-np.array(self.labeler.osg.efficiencies), label=r'$\mathtt{EW\text{-}CA}$', color='blue')
                            plt.xlabel(r'$t$',fontsize=15)
                            plt.ylabel('Inefficiency',fontsize=15)
                            # plt.title('FDR')
                            plt.legend(fontsize=10, loc='upper right')
                            plt.grid(True, which='both', linestyle='--', linewidth=0.5, zorder=1)
                            plt.tight_layout()
                            
                            # plt.axvspan(plt.xlim()[0], x_split_points[0], color='red', alpha=0.3)
                            # for i in x_split_points:
                            #     plt.axvline(x=i, color='red', linestyle='dotted', alpha=0.7, zorder=1)
                                # if i % 3 == 0: color = 'red'
                                # elif i % 3 == 1: color = 'blue'
                                # else: color = 'green'
                            plt.savefig(os.path.join(self.params.output_dir, 'inefficiencies_plot.png'))
                            pdf.savefig(bbox_inches='tight')
                            plt.close()
                        
                        
                        with PdfPages(os.path.join(self.params.output_dir, 'weights_plot.pdf')) as pdf:
                            plt.figure()
                            if isinstance(self.labeler.osg, OSGSemiAdvBanditLearner):
                                plt.plot(self.labeler.osg.weights, label=r'$\mathtt{ExAUL}$', color='green')
                            elif isinstance(self.labeler.osg, OSGAdvBanditLearner):
                                plt.plot(self.labeler.osg.weights, label=r'$\mathtt{Exp3\text{-}CA}$', color='red')
                            elif isinstance(self.labeler.osg, OSGLearner):
                                plt.plot(self.labeler.osg.weights, label=r'$\mathtt{EW\text{-}CA}$', color='blue')
                            plt.xlabel(r'$\tau$',fontsize=15)
                            plt.ylabel('Weight',fontsize=15)
                            # plt.title('FDR')
                            plt.legend(fontsize=10, loc='upper right')
                            plt.grid(True, which='both', linestyle='--', linewidth=0.5, zorder=1)
                            plt.tight_layout()
                            plt.savefig(os.path.join(self.params.output_dir, 'weights_plot.png'))
                            pdf.savefig(bbox_inches='tight')
                            plt.close()
                    except OSError as e:
                        print(f"Error saving plots: {e}")
                        print("Skipping plot saving.")
        except Exception as e:
            tqdm.write(f"[checkpoint] simulation interrupted ({type(e).__name__}: {e}); saving state.")
            self._save_checkpoint(last_full_state)
        
        x_split_points = self.x_split#[0] + self.x_split + [self.labeler.osg.T]
        shift_split_points = self.shift
        print(x_split_points)
        
        fdata = 0
        for data in chunk_log:
            if 'labeler_osg' in data and data['labeler_osg']['label'] == 0:
                fdata += 1
        ylim = (fdata / self.labeler.x_idx) if self.labeler.x_idx > 0 else self.params.alpha
        # if self.labeler.x_idx == 0:
        #     print("Warning: no labeling steps were recorded (labeler.x_idx == 0). Plot bounds default to alpha.")
                
        with PdfPages(os.path.join(self.params.output_dir, 'fdrs_plot.pdf')) as pdf:
            plt.figure()
            if isinstance(self.labeler.osg, OSGSemiAdvBanditLearner):
                plt.plot(self.labeler.osg.fdrs, label=r'$\mathtt{ExAUL}$', color='green')
            elif isinstance(self.labeler.osg, OSGAdvBanditLearner):
                plt.plot(self.labeler.osg.fdrs, label=r'$\mathtt{Exp3\text{-}CA}$', color='red')
            elif isinstance(self.labeler.osg, OSGLearner):
                plt.plot(self.labeler.osg.fdrs, label=r'$\mathtt{EW\text{-}CA}$', color='blue')
            
            plt.plot(self.labeler.osg.emp_fdrs, label=r'$\mathtt{No\text{-}CA}$', color='orange')
            plt.xlabel(r'$t$',fontsize=15)
            plt.ylabel('FDR',fontsize=15)
            if abs(ylim - self.params.alpha) < 1e-12:
                ylo = max(0.0, self.params.alpha - 0.05)
                yhi = min(1.0, self.params.alpha + 0.05)
            else:
                ylo = self.params.alpha - (ylim - self.params.alpha) * 0.9
                yhi = ylim + (ylim - self.params.alpha) * 0.9
            plt.ylim(ylo, yhi)
            # plt.title('FDR')
            # plt.axhline(y=ylim, color='k', alpha=0.7, linestyle='dotted', label='Empirical FDR upper-bound')
            plt.axhline(y=self.params.alpha, color='k', linestyle='dashed', label=r'$\alpha$='+f'{self.params.alpha}')

            plt.legend(fontsize=10, loc='upper right')
            plt.grid(True, which='both', linestyle='--', linewidth=0.5, zorder=1)
            plt.tight_layout()
            
            # plt.axvspan(plt.xlim()[0], x_split_points[0], color='red', alpha=0.3)
            # for i in x_split_points:
            #     plt.axvline(x=i, color='red', linestyle='dotted', alpha=0.7, zorder=1)
                # if i % 3 == 0: color = 'red'
                # elif i % 3 == 1: color = 'blue'
                # else: color = 'green'
            plt.savefig(os.path.join(self.params.output_dir, 'fdrs_plot.png'))
            pdf.savefig(bbox_inches='tight')
            plt.close()
        
        with PdfPages(os.path.join(self.params.output_dir, 'inefficiencies_plot.pdf')) as pdf:
            plt.figure()
            if isinstance(self.labeler.osg, OSGSemiAdvBanditLearner):
                plt.plot(1-np.array(self.labeler.osg.efficiencies), label=r'$\mathtt{ExAUL}$', color='green')
            elif isinstance(self.labeler.osg, OSGAdvBanditLearner):
                plt.plot(1-np.array(self.labeler.osg.efficiencies), label=r'$\mathtt{Exp3\text{-}CA}$', color='red')
            elif isinstance(self.labeler.osg, OSGLearner):
                plt.plot(1-np.array(self.labeler.osg.efficiencies), label=r'$\mathtt{EW\text{-}CA}$', color='blue')
            plt.xlabel(r'$t$',fontsize=15)
            plt.ylabel('Inefficiency',fontsize=15)
            # plt.title('FDR')
            plt.legend(fontsize=10, loc='upper right')
            plt.grid(True, which='both', linestyle='--', linewidth=0.5, zorder=1)
            plt.tight_layout()
            
            # plt.axvspan(plt.xlim()[0], x_split_points[0], color='red', alpha=0.3)
            # for i in x_split_points:
            #     plt.axvline(x=i, color='red', linestyle='dotted', alpha=0.7, zorder=1)
                # if i % 3 == 0: color = 'red'
                # elif i % 3 == 1: color = 'blue'
                # else: color = 'green'
            plt.savefig(os.path.join(self.params.output_dir, 'inefficiencies_plot.png'))
            pdf.savefig(bbox_inches='tight')
            plt.close()
        
        
        with PdfPages(os.path.join(self.params.output_dir, 'weights_plot.pdf')) as pdf:
            plt.figure()
            if isinstance(self.labeler.osg, OSGSemiAdvBanditLearner):
                plt.plot(self.labeler.osg.weights, label=r'$\mathtt{ExAUL}$', color='green')
            elif isinstance(self.labeler.osg, OSGAdvBanditLearner):
                plt.plot(self.labeler.osg.weights, label=r'$\mathtt{Exp3\text{-}CA}$', color='red')
            elif isinstance(self.labeler.osg, OSGLearner):
                plt.plot(self.labeler.osg.weights, label=r'$\mathtt{EW\text{-}CA}$', color='blue')
            plt.xlabel(r'$\tau$',fontsize=15)
            plt.ylabel('Weight',fontsize=15)
            # plt.title('FDR')
            plt.legend(fontsize=10, loc='upper right')
            plt.grid(True, which='both', linestyle='--', linewidth=0.5, zorder=1)
            plt.tight_layout()
            plt.savefig(os.path.join(self.params.output_dir, 'weights_plot.png'))
            pdf.savefig(bbox_inches='tight')
            plt.close()
            
            
        # print(result)
        result = []
        data_log = []
        for idx, data in enumerate(chunk_log):
            if 'labeler_osg' in data:
                logprobs = chunk_log[idx-1]['chat_bot']['logprobs'] if 'chat_bot' in chunk_log[idx-1] and 'logprobs' in chunk_log[idx-1]['chat_bot'] else None
                label = data['labeler_osg']['label']
                t = data['labeler_osg']['t']
                f_std = data['labeler_osg']['score']
                question = chunk_log[idx-2]['user']['messages'][-1].content
                generation = chunk_log[idx-1]['chat_bot']['messages'][-1].content
                context = chunk_log[idx-2]['user']['context']
                samples = chunk_log[idx-1]['chat_bot']['samples']
                sample_score = chunk_log[idx-1]['chat_bot']['sample_score']
                data_log.append(
                    {
                        't': t,
                        'question': question,
                        'generated_sequence': generation,
                        'context': context,
                        'label': label,
                        'samples': samples,
                        'logprobs': logprobs if logprobs is not None else None,
                        'sample_score': sample_score,
                    }
                )
        for idx, data in enumerate(chunk_log):
            if 'user' in data:
                data['user']['messages'] = [m.content for m in data['user']['messages']]
                data['user']['user_messages'] = [m.content for m in data['user']['user_messages']]
                data['user']['memory'] = [m.content for m in data['user']['memory']]
                data['user']['user_memory'] = [m.content for m in data['user']['user_memory']]
            if 'chat_bot' in data:
                data['chat_bot']['messages'] = [m.content for m in data['chat_bot']['messages']]
                data['chat_bot']['memory'] = [m.content for m in data['chat_bot']['memory']]
            if 'labeler_osg' in data:
                data['labeler_osg']['user_messages'] = [m.content for m in data['labeler_osg']['user_messages']]
                data['labeler_osg']['user_memory'] = [m.content for m in data['labeler_osg']['user_memory']]
                data['labeler_osg']['score'] = float(data['labeler_osg']['score'])
                data['labeler_osg']['index'] = int(data['labeler_osg']['index'])
            result.append(data)
            
            
        json.dump(data_log, open(os.path.join(self.params.output_dir, f'dataset.json'), 'w'), indent=4)
        json.dump(result, open(os.path.join(self.params.output_dir, 'state_log.json'), 'w'), indent=4)
        json.dump(self.labeler.osg.fdrs, open(os.path.join(self.params.output_dir, 'fdrs.json'), 'w'), indent=4)
        json.dump(self.labeler.osg.efficiencies, open(os.path.join(self.params.output_dir, 'efficiencies.json'), 'w'), indent=4)
        json.dump(self.labeler.osg.weights.tolist(), open(os.path.join(self.params.output_dir, 'weights.json'), 'w'), indent=4)
        # json.dump(x_split_points, open(os.path.join(self.params.output_dir, 'x_split_points.json'), 'w'), indent=4)
        # json.dump(shift_split_points, open(os.path.join(self.params.output_dir, 'shift_split_points.json'), 'w'), indent=4)

        if self.labeler.x_idx >= self.T:
            self._clear_checkpoint()
        
        
    
    
    
    
    def my_chat_bot(self, messages):
        system_message = {
            "role": "system",
            "content": "You are a helpful AI assistant for question answering. Answer the question from the user in one or two sentences. \
                **Do not ask questions** — your role is to **answer**, not inquire.\n\nDo not say 'I don't know' or 'I can't answer that'.",
        }
        messages = [system_message] + messages
        # print('여기1')
        completion = openai.chat.completions.create(
            messages=messages, model=self.params.model,
            # seed=123,
            max_completion_tokens=500,
            temperature=0.0,
            # logprobs=True if self.params.confidence_score == "fM1" else False,
        )
        # print('여기2')
        if self.params.confidence_score == "fM2":
            # samples = openai.chat.completions.create(
            #     messages=messages, model=self.params.model,
            #     # seed=123,
            #     max_tokens=500,
            #     temperature=1.0,
            #     # logprobs=True,
            #     n=10,  # Generate 10 samples
            # )
            samples = []
            for _ in range(10):
                sample = openai.chat.completions.create(
                    messages=messages,
                    model=self.params.model,
                    max_completion_tokens=500,
                    temperature=1.0,
                    # logprobs=True,
                )
                samples.append(sample.choices[0].message.model_dump())
            
        # print('여기3')
        return completion.choices[0].message.model_dump(), \
            [token.logprob for token in completion.choices[0].logprobs.content] if completion.choices[0].logprobs else None, \
            samples if self.params.confidence_score == "fM2" else None
            # samples.choices if self.params.confidence_score == "fM2" else None

    def chat_bot_node(self, state):
        # messages = state["messages"]
        messages = state.get("memory", [])

        # Convert from LangChain format to the OpenAI format, which our chatbot function expects.
        messages = [convert_message_to_dict(m) for m in messages]
        # Call the chat bot
        chat_bot_response, logprobs, chat_bot_samples = self.my_chat_bot(messages)
        # try:
        #     chat_bot_response, logprobs = self.my_chat_bot(messages)
        # except openai.BadRequestError as e:
        #     print(f"Error: {e}")
        #     print("SHIFTING CONTEXT so that EMPTY MEMORY")
            
        #     return{
        #         "messages": [AIMessage(content="SHIFT")],
        #         # "user_memory": state["user_memory"]
        #     }
        #     # # If the chat bot fails. May not happen!
        #     # messages = messages[-3:]
        #     # state["memory"] = messages
        #     # state["user_memory"] = state["user_memory"][-3:]
        #     # chat_bot_response, logprobs = self.my_chat_bot(messages)
        
        if self.params.confidence_score == "fM2":
            sample_contents = [s["content"] for s in chat_bot_samples]
            sample_scores = [self.sample_score(chat_bot_response["content"], s) for s in sample_contents]
            sample_scores = sum(sample_scores) / len(sample_scores) / 10.0

        memory = state.get("memory", []) + [AIMessage(content=chat_bot_response["content"])]
        # Respond with an AI Message
        return {
            "messages": [AIMessage(content=chat_bot_response["content"])],
            # "generation": chat_bot_response["content"],
            "logprobs": logprobs if logprobs is not None else None,
            "memory": memory,
            "samples": sample_contents if self.params.confidence_score == "fM2" else None,
            "sample_score": sample_scores if self.params.confidence_score == "fM2" else None,
        }

    def simulated_user_node(self, state):
        # messages = state["user_messages"]
        messages = state.get("user_memory", [])
        # messages = messages[-5:]
        if (len(messages) > 1 and messages[-1].content == "SHIFT") or len(messages) >= self.max_turn:
            state["memory"] = [] #  empty memory.
            state["user_memory"] = [] #  empty memory.
            new_messages = state.get("user_memory", [])
            
            
            # INFO: context-chunked
            # if len(self.contexts) == 0:
            #     init_data = next(self.iterd)
            #     self.x_split.append(self.labeler.x_idx)
            #     self.title = init_data["title"]
            #     self.contexts = init_data["contexts"]
            
            # self.context = self.contexts.pop(0)
            
            
            init_data = next(self.iterd)
            self._iterd_idx += 1

            if self.id != init_data["id"]:
                self.x_split.append(self.labeler.x_idx)
            self.shift.append(self.labeler.x_idx)
            self.context = init_data["context"]
            self.title = init_data["title"]
            self.id = init_data["id"]
            
        else:
            # Swap roles of messages
            new_messages = _swap_roles(messages)
        
        
        prompt = self.prompt.partial(instructions=self.instructions, context=self.context)
        model = ChatOpenAI(model="gpt-5.4-mini", temperature=0.0, max_tokens=500)

        self.simulated_user = prompt | model
        
        # Call the simulated user
        response = self.simulated_user.invoke({"messages": new_messages})
        # This response is an AI message - we need to flip this to be a human message
        
        memory = state.get("memory", []) + [HumanMessage(content=response.content)]
        user_memory = state.get("user_memory", []) + [HumanMessage(content=response.content)]
        return {
            "messages": [HumanMessage(content=response.content)],
            "user_messages": [HumanMessage(content=response.content)],
            "context": self.context,
            "memory": memory,
            "user_memory": user_memory,
        }
    def sample_score(self, generated_sequence: str, sample: str) -> float:
        
        system = (
            "You are an expert in Natural Language Inference (NLI). "
            "Rate how strongly the generated_sequence entails the sample on a scale from 0 to 10."
            "Return only the integer score (0–10). 0 (complete contradiction) to 10 (full entailment)"
        )
        user = (
            f"GENERATED_SEQUENCE: {generated_sequence}\n\n"
            f"SAMPLE: {sample}\n\n"
            "Score:"
        )
        
        resp = self.client.chat.completions.create(
            model="gpt-5.4-nano",
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user}
            ],
            temperature=0.0,
            max_completion_tokens=10
        )
        return int(resp.choices[0].message.content.strip())

    def should_continue(self, state):
        try:
            messages = state["messages"]
            last = messages[-1].content if messages else None
            if self._dbg_enabled and "should_continue" not in self._dbg_printed:
                print(f"[debug] should_continue last={last!r}")
                self._dbg_printed.add("should_continue")
            return "shift" if last == "SHIFT" else "continue"
        except Exception:
            if self._dbg_enabled:
                print("[debug] should_continue crashed")
            raise
        
    def should_continue_labeler(self, state):
        if self._dbg_enabled and "should_continue_labeler" not in self._dbg_printed:
            print(f"[debug] should_continue_labeler x_idx={self.labeler.x_idx} T={self.T}")
            self._dbg_printed.add("should_continue_labeler")
        return "end" if self.labeler.x_idx >= self.T else "continue"

    def _checkpoint_path(self):
        return os.path.join(self.params.output_dir, CHECKPOINT_FILENAME)

    def _save_checkpoint(self, last_state):
        payload = {
            'osg': self.labeler.osg,
            'labeler_x_idx': self.labeler.x_idx,
            'x_split': self.x_split,
            'shift': self.shift,
            'id': self.id,
            'title': self.title,
            'context': self.context,
            'iterd_idx': self._iterd_idx,
            'chunk_log': self._chunk_log,
            'last_state': last_state,
        }
        path = self._checkpoint_path()
        tmp = path + '.tmp'
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(tmp, 'wb') as f:
                pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, path)
        except Exception as e:
            tqdm.write(f"[checkpoint] save failed: {e}")

    def _load_checkpoint(self):
        path = self._checkpoint_path()
        if not os.path.exists(path):
            return None
        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"[checkpoint] load failed: {e}")
            return None

    def _clear_checkpoint(self):
        path = self._checkpoint_path()
        if os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass










class Labeler:
    PROMPT = inspect.cleandoc(
        """\
        You are an expert in Answer Verification. Your task is to determine whether the generated answer is correct, given the context, the conversation history, and the question–answer pair.
        
        Please provide a score with a score of 0 or 1, where:
        1 = correct
        0 = incorrect
        
        The context: {context}
        
        The question: {question}
        The generated answer: {answer}

        Once again, provide only a single score — either 0 or 1.
        Score:"""
    )
    
    def __init__(self, model, params):
        self.params = params
        
        self.max_tokens=500
        self.temperature=0
        self.stop=None
        self.seed=123
        self.tools=None
        self.logprobs=False
        self.top_logprobs=None
        # self.model = "o4-mini"
        # Default to the generation model unless a separate labeler model is provided.
        self.model = getattr(params, "labeler_model", None) or getattr(params, "model", None) or "gpt-4o"
        
        self.x_idx = 0
        
        self.osg = model
        
        self.client = openai.OpenAI(api_key=_get_openai_api_key())
        
    def __call__(
        self,
        state,
    ):
        # print("여기야 여기", self.PROMPT.format(
        #     history=state["messages"][:-2] if len(state["messages"]) > 2 else None,
        #     context=state["context"],
        #     question=state["messages"][-1].content,
        #     answer=state["messages"][-1].content
        # ))
        # print("Here debugging")
        # print(state["memory"])
        
        messages = [{"role": "user", "content": self.PROMPT.format(
            # history=state["memory"][:-2] if len(state["memory"]) > 2 else None,
            context=state["context"],
            question=state["messages"][-2].content,  # [-1] is the chatbot answer, [-2] the user question
            answer=state["messages"][-1].content
        )}]
        
        params = {
            "model": self.model,
            "messages": messages,
            "max_completion_tokens": self.max_tokens,
            # "stop": self.stop,
            # "seed": self.seed,
            # "logprobs": self.logprobs,
            # "top_logprobs": self.top_logprobs,
        }
        if self.tools is not None:
            params["tools"] = self.tools
        
        try:
            completion = self.client.chat.completions.create(**params)
        except openai.BadRequestError as e:
            print(f"Error: {e}")
            print("Slicing the messages to the previous message.")
            # If the chat bot fails. May not happen! #FIXME
            messages = [{"role": "user", "content": self.PROMPT.format(
                # history=state["memory"][-6:-2] if len(state["memory"]) > 2 else None,
                context=state["context"],
                question=state["messages"][-2].content,  # [-1] is the chatbot answer, [-2] the user question
                answer=state["messages"][-1].content
            )}]
            
            params = {
                "model": self.model,
                "messages": messages,
                "max_completion_tokens": self.max_tokens,
                "temperature": self.temperature,
                # "stop": self.stop,
                # "seed": self.seed,
                # "logprobs": self.logprobs,
                # "top_logprobs": self.top_logprobs,
            }
            if self.tools is not None:
                params["tools"] = self.tools
            
            completion = self.client.chat.completions.create(**params)
        
        response = completion.choices[0].message.content
        label = int(response.strip()) if response.strip() in ["0", "1"] else None
        
        attempts = 0
        while label is None and attempts < 3:
            print(f"Retrying labeling... Attempt {attempts + 1}")
            attempts += 1
            params["messages"].append({"role": "assistant", "content": completion.choices[0].message.content})
            retry_message = {"role": "user", "content": "Your response is not valid. Please provide a score of only 0 or 1, try again. Score:"}
            params["messages"].append(retry_message)
            completion = self.client.chat.completions.create(**params)
            response = completion.choices[0].message.content
            label = int(response.strip()) if response.strip() in ["0", "1"] else None
            
        if label is None:
            raise ValueError(f"Invalid response after 10 attempts: {response}")
        

        if self.params.confidence_score == "fM1":
            score = np.exp(np.array(state["logprobs"]).mean())
        elif self.params.confidence_score == "fM2":
            score = state["sample_score"]
        index = self.osg.predict()
        self.osg.update(score, index, label)
        self.x_idx += 1
        
        # # print("step / T:", f"{self.osg.t} / {self.osg.T}")
        # state["messages"] = state["messages"][-1]
        result_mesage = AIMessage(content=state["messages"][-1].content) if self.osg.tau_h[index] <= score \
                        else AIMessage(content="I don't know.. This is out of my scope. Can you ask me something else?")
        user_memory = state.get("user_memory", []) + [result_mesage]
        
        return {
            "user_messages": [result_mesage],
            "user_memory": user_memory,
            "label": label,
            "IDK_flag": False if self.osg.tau_h[index] <= score else True,
            "t": self.x_idx,
            "score": score,
            "index": index,
        }        

def _swap_roles(messages):
    new_messages = []
    for m in messages:
        if isinstance(m, AIMessage):
            new_messages.append(HumanMessage(content=m.content))
        else:
            new_messages.append(AIMessage(content=m.content))
    return new_messages



class State(TypedDict):
    messages: Annotated[list, add_messages]
    user_messages: Annotated[list, add_messages]
    memory: list[HumanMessage | AIMessage]
    user_memory: list[HumanMessage | AIMessage]
    logprobs: list[float]
    label: int
    generation: str
    context: str
    IDK_flag: bool
    t: input
    score: float
    index: int
    samples: list[str]
    sample_score: float
