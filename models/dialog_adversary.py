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
from models.dialog import _get_openai_api_key
import json


class DialogAdversaryLearner:
    system_prompt = inspect.cleandoc(
        """\
        You are an adversary engaged in a game with a learner model. Your goal is to expose the learner's weaknesses.

        Below is the list of candidate questions with the decision history you can ask:
        {history}

        Current Status:
        - the current step: {step}
        - The total mean of incorrect answers not rejected (IDK==False and label==0) over steps: {total_incorrect_not_rejected}
        - The total mean of correct answers rejected (IDK==True and label==1) over steps: {total_correct_rejected}

        {instructions}

        When you want to find more proper questions for maximizing your objective, say 'SHIFT' to get a new question.
        You must decide this **yourself**  do **not** ask whether to 'SHIFT', simply respond with 'SHIFT' by yourself when needed.
        """
    )

    instructions = inspect.cleandoc(
        """\
        **Your Objective:**
        You are an adversary trying to **maximize the total mean of incorrect answer not rejected (IDK==False and label==0)** by the learner.
        However, you must know that the learner trains a threshold to reject incorrect answers based on your questions and their confidence score of answers,
        thus you need to be strategic in asking questions, i.e., you should also **maximize the total mean of correct answers rejected (IDK==True and label==1)** by the learner.
        E.g., if the learner starts to reject incorrect answers more often, you should also try to ask other questions that may fool the learner.

        Keep in mind that:
        - The learner updates its rejection threshold over time.
        - Look for IDK patterns in the history list.
        - Select questions that are likely to maximize your objective.
        
        **Output Format:**
        - If you select a question, respond with **ONLY the index number** (e.g., 5).
        - Do not write any other explanation or text.
        """
    )
# class DialogAdversaryLearner:
#     system_prompt = inspect.cleandoc(
#         """\
#         You are an adversary playing against a learner that answers questions and sometimes says "I don't know" (IDK). Your goal is to hurt the learner by maximizing the objectives defined below.

#         === Candidate questions (history you may reuse) ===
#         {history}

#         === Current Status ===
#         - step t                                                                                     : {step}
#         - the total number of incorrect answers NOT rejected (IDK==False and label==0) over steps    : {incorrect_not_rejected_rate}   (PRIMARY objective — maximize)
#         - the total number of correct answers REJECTED (IDK==True and label==1) over steps           : {correct_rejected_rate}         (SECONDARY objective — also maximize)

#         {instructions}

#         === SHIFT policy (use it freely when it helps) ===
#         - 'SHIFT' pulls a brand-new question that is not yet in the history. It is a NORMAL action, not a last resort — use it whenever the history no longer gives you a clearly useful pick, because the learner needs varied probes to keep updating its threshold in a direction you can exploit.
#         - Good reasons to SHIFT include any of:
#             (a) the history is empty or very small and more diversity is needed to let the learner actually train,
#             (b) no history item looks useful for the PRIMARY or as a fooling move right now,
#             (c) a few recent attempts on the same history item have not changed IDK_flag in a useful direction (diminishing returns),
#             (d) the current pool has drifted and fresh signal would help more than reusing old items.
#         - That said, do NOT shift on EVERY step. If a history item is clearly working — either slipping through to advance the PRIMARY, or being falsely rejected to fool the learner — reuse it for a few steps before shifting.
#         - Do NOT ask whether to shift; just output the literal token SHIFT alone when any of the conditions above hold.
#         """
#     )

#     instructions = inspect.cleandoc(
#         """\
#         === Your objective ===
#         Your true goal is the PRIMARY objective: maximize the count of *incorrect answers the learner failed to reject* (label==0 AND IDK_flag==False).

#         The SECONDARY objective — maximizing the count of *correct answers the learner rejected* (label==1 AND IDK_flag==True) — is a TOOL for fooling the learner. By feeding correct questions that the learner wrongly rejects, you push the learner's rejection threshold downward, which in turn lets more label==0 answers slip through and boosts the PRIMARY objective.

#         Keep in mind that:
#         - The learner updates its rejection threshold over time.
#         - Look for IDK patterns in the history list.
#         - Select questions that are likely to maximize your PRIMARY objective — either directly, or indirectly by using SECONDARY-style picks to fool the learner into lowering its threshold.

#         === Output format (STRICT) ===
#         - To reuse a history question: respond with **ONLY its integer index** (e.g., 5). No explanation, no punctuation, no other text.
#         - To request a fresh question: respond with **ONLY the token** SHIFT. No explanation, no punctuation, no other text.
#         - Any other output is invalid.
#         """
#     )
    # system_prompt = inspect.cleandoc(
    #     """\
    #     You are an adversary who continuously asks questions from the given history. The following is the history for question answering:
        
    #     History (label 0 indicates incorrect generated sequence (answer), and label 1 indicates correct generated sequence):
    #     {history}
        
    #     The number of incorrect answers not rejected (not IDK) so far: {total_incorrect_not_rejected}
    #     The number of correct answers rejected (IDK) so far: {total_correct_rejected}

    #     {instructions}

    #     When you cannot find proper questions in the history for maximizing your objective, say 'SHIFT' to get a new question.
    #     You must decide this **yourself** — do **not** ask whether to 'SHIFT', simply respond with 'SHIFT' by yourself when needed.
    #     """
    #     # When you are finished with the conversation, respond with a single word 'FINISHED'.
    #     # However, do **not** respond with 'FINISHED' unless a **serious problem** occurs that prevents you from continuing the conversation.
    #     # """
    # )
    # instructions = inspect.cleandoc(
    #     """\
    #     You are an adversary trying to maximize the number of incorrect answer not rejected (not IDK) by the learner.
    #     However, you must know that the learner trains a threshold to reject incorrect answers based on your questions and their confidence score of answers,
    #     thus you need to be strategic in asking questions, i.e., you should also maximize the number of correct answers rejected (IDK) by the learner.
        
    #     Based on the given history, you pick a question to ask the learner.
    #     You must pick the index of the question from the provided list of questions from the history.
    #     """
    # )
    
    def __init__(self, model, params=None):
        # self.labeler = Labeler(model, params)
        self.osg = model
        
        self.params = params
        self.T = params.T
        self.x_split = []
        self.shift = []
        self.max_history = 100
        self.client = openai.OpenAI(api_key=_get_openai_api_key())
        
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
            ]
        )
        self.rd_iter_idx = 0
        self.idx = 0
        self.history = []
        self.total_incorrect_not_rejected = 0
        self.total_correct_rejected = 0
        self.total_incorrect_not_rejected_list = []
        self.total_correct_rejected_list = []

        self._prev_data_idx = None
        self._consec_same = 0
        self._max_consec_same = int(getattr(params, "max_consec_same_idx", 10))
        
        graph_builder = StateGraph(State)
        graph_builder.add_node("user", self.simulated_user_node)
        graph_builder.add_node("chat_bot", self.chat_bot_node)
        # graph_builder.add_node("labeler_osg", self.labeler)
        # Every response from  your chat bot will automatically go to the
        # simulated user
        
        # graph_builder.add_edge("chat_bot", "labeler_osg")
        graph_builder.add_conditional_edges(
            "user",
            self.should_continue,
            {
                "shift": "user",
                "continue": "chat_bot",
            },
        )
        graph_builder.add_conditional_edges(
            "chat_bot",
            self.should_continue_labeler,
            {
                "end": END,
                "continue": "user",
            },
        )
        
        graph_builder.add_edge(START, "user")
        self.simulation = graph_builder.compile()
        
        
    def simulate(self, rd, updated_params=None):
        # self.params = updated_params if updated_params else self.params
        if 'sample_score' in rd.column_names:
            before = len(rd)
            rd = rd.filter(lambda x: x['sample_score'] < 1.0)
            after = len(rd)
            if before != after:
                print(f"[filter] dropped {before - after} rows with sample_score == 1 (kept {after}/{before})")
        self.rd = rd
        
        # self.iterd = iter(rd)
        
        # init_data = next(self.iterd)
        # self.contexts = init_data["contexts"]
        # self.context = self.contexts.pop(0)
        
        # self.id = init_data["id"]
        # self.title = init_data["title"]
        # self.context = init_data["context"]
        
        chunk_log = []
        pbar = tqdm(total=self.T, desc="Executing states")
        
        # try:
        for chunk in self.simulation.stream({"messages": []}, config={"recursion_limit": 50000}):
            
                
            if END not in chunk:
                tqdm.write(str(chunk))
                tqdm.write("---")
                
                
                chunk_log.append(chunk)
            
            if "chat_bot" in chunk:
                tqdm.write("="*30)
                pbar.update(1)
                
                fdata = 0
                for data in chunk_log:
                    if 'chat_bot' in data and data['chat_bot']['label'] == 0:
                        fdata += 1
                ylim = fdata / self.idx
                        
                try:
                    with PdfPages(os.path.join(self.params.output_dir, 'fdrs_plot.pdf')) as pdf:
                        plt.figure()
                        if isinstance(self.osg, OSGSemiAdvBanditLearner):
                            plt.plot(self.osg.fdrs, label=r'$\mathtt{ExAUL}$', color='green')
                        elif isinstance(self.osg, OSGAdvBanditLearner):
                            plt.plot(self.osg.fdrs, label=r'$\mathtt{Exp3\text{-}CA}$', color='red')
                        elif isinstance(self.osg, OSGLearner):
                            plt.plot(self.osg.fdrs, label=r'$\mathtt{EW\text{-}CA}$', color='blue')
                        
                        plt.plot(self.osg.emp_fdrs, label=r'$\mathtt{No\text{-}CA}$', color='orange')
                        plt.xlabel(r'$t$',fontsize=15)
                        plt.ylabel('FDR',fontsize=15)
                        plt.ylim(self.params.alpha - (ylim-self.params.alpha)*0.9, ylim + (ylim-self.params.alpha)*0.9)
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
                        if isinstance(self.osg, OSGSemiAdvBanditLearner):
                            plt.plot(1-np.array(self.osg.efficiencies), label=r'$\mathtt{ExAUL}$', color='green')
                        elif isinstance(self.osg, OSGAdvBanditLearner):
                            plt.plot(1-np.array(self.osg.efficiencies), label=r'$\mathtt{Exp3\text{-}CA}$', color='red')
                        elif isinstance(self.osg, OSGLearner):
                            plt.plot(1-np.array(self.osg.efficiencies), label=r'$\mathtt{EW\text{-}CA}$', color='blue')
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
                        if isinstance(self.osg, OSGSemiAdvBanditLearner):
                            plt.plot(self.osg.weights, label=r'$\mathtt{ExAUL}$', color='green')
                        elif isinstance(self.osg, OSGAdvBanditLearner):
                            plt.plot(self.osg.weights, label=r'$\mathtt{Exp3\text{-}CA}$', color='red')
                        elif isinstance(self.osg, OSGLearner):
                            plt.plot(self.osg.weights, label=r'$\mathtt{EW\text{-}CA}$', color='blue')
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
        # except:
        #     pass
        
        x_split_points = self.x_split#[0] + self.x_split + [self.labeler.osg.T]
        shift_split_points = self.shift
        print(x_split_points)
        
        fdata = 0
        for data in chunk_log:
            if 'chat_bot' in data and data['chat_bot']['label'] == 0:
                fdata += 1
        ylim = fdata / self.idx
                
        with PdfPages(os.path.join(self.params.output_dir, 'fdrs_plot.pdf')) as pdf:
            plt.figure()
            if isinstance(self.osg, OSGSemiAdvBanditLearner):
                plt.plot(self.osg.fdrs, label=r'$\mathtt{ExAUL}$', color='green')
            elif isinstance(self.osg, OSGAdvBanditLearner):
                plt.plot(self.osg.fdrs, label=r'$\mathtt{Exp3\text{-}CA}$', color='red')
            elif isinstance(self.osg, OSGLearner):
                plt.plot(self.osg.fdrs, label=r'$\mathtt{EW\text{-}CA}$', color='blue')
            
            plt.plot(self.osg.emp_fdrs, label=r'$\mathtt{No\text{-}CA}$', color='orange')
            plt.xlabel(r'$t$',fontsize=15)
            plt.ylabel('FDR',fontsize=15)
            plt.ylim(self.params.alpha - (ylim-self.params.alpha)*0.9, ylim + (ylim-self.params.alpha)*0.9)
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
            if isinstance(self.osg, OSGSemiAdvBanditLearner):
                plt.plot(1-np.array(self.osg.efficiencies), label=r'$\mathtt{ExAUL}$', color='green')
            elif isinstance(self.osg, OSGAdvBanditLearner):
                plt.plot(1-np.array(self.osg.efficiencies), label=r'$\mathtt{Exp3\text{-}CA}$', color='red')
            elif isinstance(self.osg, OSGLearner):
                plt.plot(1-np.array(self.osg.efficiencies), label=r'$\mathtt{EW\text{-}CA}$', color='blue')
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
            if isinstance(self.osg, OSGSemiAdvBanditLearner):
                plt.plot(self.osg.weights, label=r'$\mathtt{ExAUL}$', color='green')
            elif isinstance(self.osg, OSGAdvBanditLearner):
                plt.plot(self.osg.weights, label=r'$\mathtt{Exp3\text{-}CA}$', color='red')
            elif isinstance(self.osg, OSGLearner):
                plt.plot(self.osg.weights, label=r'$\mathtt{EW\text{-}CA}$', color='blue')
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
            if 'chat_bot' in data:
                # logprobs = chunk_log[idx-1]['chat_bot']['logprobs']
                label = data['chat_bot']['label']
                t = data['chat_bot']['t']
                # f_std = data['labeler_osg']['score']
                question = chunk_log[idx-1]['user']['messages'][-1].content
                generation = chunk_log[idx]['chat_bot']['messages'][-1].content
                data_idx = chunk_log[idx-1]['user']['data_idx']
                history = chunk_log[idx]['chat_bot']['history']
                # context = chunk_log[idx-2]['user']['context']
                # samples = chunk_log[idx]['chat_bot']['samples']
                # sample_score = chunk_log[idx]['chat_bot']['sample_score']
                data_log.append(
                    {
                        't': t,
                        'question': question,
                        'generated_sequence': generation,
                        # 'context': context,
                        'label': label,
                        'data_idx': data_idx,
                        'history': history,
                        # 'samples': samples,
                        # 'logprobs': logprobs,
                        # 'sample_score': sample_score,
                    }
                )
        for idx, data in enumerate(chunk_log):
            if 'user' in data:
                data['user']['messages'] = [m.content for m in data['user']['messages']]
                # data['user']['user_messages'] = [m.content for m in data['user']['user_messages']]
                # data['user']['memory'] = [m.content for m in data['user']['memory']]
                # data['user']['user_memory'] = [m.content for m in data['user']['user_memory']]
            if 'chat_bot' in data:
                data['chat_bot']['messages'] = [m.content for m in data['chat_bot']['messages']]
                # data['chat_bot']['memory'] = [m.content for m in data['chat_bot']['memory']]
                data['chat_bot']['score'] = float(data['chat_bot']['score'])
                data['chat_bot']['index'] = int(data['chat_bot']['index'])
            # if 'labeler_osg' in data:
            #     data['labeler_osg']['user_messages'] = [m.content for m in data['labeler_osg']['user_messages']]
            #     data['labeler_osg']['user_memory'] = [m.content for m in data['labeler_osg']['user_memory']]
            result.append(data)
            
            
        json.dump(data_log, open(os.path.join(self.params.output_dir, f'dataset.json'), 'w'), indent=4)
        json.dump(result, open(os.path.join(self.params.output_dir, 'state_log.json'), 'w'), indent=4)
        json.dump(self.osg.fdrs, open(os.path.join(self.params.output_dir, 'fdrs.json'), 'w'), indent=4)
        json.dump(self.osg.efficiencies, open(os.path.join(self.params.output_dir, 'efficiencies.json'), 'w'), indent=4)
        json.dump(self.osg.weights.tolist(), open(os.path.join(self.params.output_dir, 'weights.json'), 'w'), indent=4)
        # json.dump(x_split_points, open(os.path.join(self.params.output_dir, 'x_split_points.json'), 'w'), indent=4)
        # json.dump(shift_split_points, open(os.path.join(self.params.output_dir, 'shift_split_points.json'), 'w'), indent=4)
        
        

    def chat_bot_node(self, state):
        # messages = state["messages"]
        messages = state.get("messages", [])


        # Convert from LangChain format to the OpenAI format, which our chatbot function expects.
        # messages = [convert_message_to_dict(m) for m in messages]
        # Call the chat bot
        # chat_bot_response, logprobs, chat_bot_samples = self.my_chat_bot(messages)
        data_idx = state["data_idx"]
        
        if self.params.confidence_score == "fM2":
            score = self.rd['sample_score'][data_idx]
            # sample_scores = [self.sample_score(chat_bot_response["content"], sample) for sample in [choice.message.content for choice in chat_bot_samples]]
            # sample_scores = sum(sample_scores) / len(sample_scores) / 10.0
        # self.rd_iter_idx += 1
            
        # memory = state.get("memory", []) + [AIMessage(content=chat_bot_response["content"])]
        # Respond with an AI Message
        
        
        label = int(self.rd['label'][data_idx])
        index = self.osg.predict()
        self.osg.update(score, index, label)
        self.idx += 1
        
        generated_sequence = self.rd['generated_sequence'][data_idx] if self.osg.tau_h[index] <= score \
                            else "I don't know.. This is out of my scope."
        # if state["is_shift"]:
        history = {
            "index": data_idx,
            "question": self.rd['question'][data_idx],
            "generated_sequence": self.rd['generated_sequence'][data_idx],
            "label": label,
            "IDK_flag": False if self.osg.tau_h[index] <= score else True,
        }
        self.history.append(history)
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
        
        if label == 0 and self.osg.tau_h[index] <= score:
            self.total_incorrect_not_rejected += 1
            self.total_incorrect_not_rejected_list.append(self.total_incorrect_not_rejected/self.idx)
        if label == 1 and self.osg.tau_h[index] > score:
            self.total_correct_rejected += 1
            self.total_correct_rejected_list.append(self.total_correct_rejected/self.idx)
        
        tqdm.write(f"Total incorrect answers not rejected so far: {self.total_incorrect_not_rejected}"
                   f" | Total correct answers rejected so far: {self.total_correct_rejected}")
        return {
            "messages": [AIMessage(content=generated_sequence)],
            # "generation": chat_bot_response["content"],
            # "logprobs": logprobs,
            # "memory": memory,
            "label": label,
            "IDK_flag": False if self.osg.tau_h[index] <= score else True,
            "t": self.idx,
            "score": score,
            "index": index,
            # Snapshot: returning self.history itself lets later appends leak into this step's log.
            "history": list(self.history),

            # "samples": [choice.message.content for choice in chat_bot_samples] if self.params.confidence_score == "fM2" else None,
            # "sample_score": sample_scores if self.params.confidence_score == "fM2" else None,
        }

    def simulated_user_node(self, state):
        # messages = state["user_messages"]
        messages = state.get("messages", [])
        
        
        if len(self.history) == 0  or (len(messages) > 1 and messages[-1].content == "SHIFT"):
            # state["memory"] = [] #  empty memory.
            # state["user_memory"] = [] #  empty memory.

            # new_messages = state.get("memory", [])
            response = self.rd['question'][self.rd_iter_idx]
            # memory = state.get("memory", []) + [HumanMessage(content=response)]

            self.rd_iter_idx += 1
            self._prev_data_idx = None
            self._consec_same = 0

            return {
                "messages": [HumanMessage(content=response)],
                # "user_messages": [HumanMessage(content=response)],
                # "history": self.history,
                # "memory": memory,
                "is_shift": True,
                "data_idx": self.rd_iter_idx - 1,
                # "user_memory": memory,
            }
            
        inr_rate = (self.total_incorrect_not_rejected / self.idx) if self.idx > 0 else 0.0
        cr_rate  = (self.total_correct_rejected      / self.idx) if self.idx > 0 else 0.0

        prompt = self.prompt.partial(
            instructions=self.instructions,
            history=str(self.history),
            step=self.idx,
            total_incorrect_not_rejected=str(self.total_incorrect_not_rejected),
            total_correct_rejected=str(self.total_correct_rejected),
            # incorrect_not_rejected_rate=f"{inr_rate:.4f}",
            # correct_rejected_rate=f"{cr_rate:.4f}",
        )  # INFO: history + running primary/secondary rates
        model = ChatOpenAI(model="gpt-5.4-mini", temperature=0.0, max_tokens=10)

        self.simulated_user = prompt | model
        
        # Call the simulated user
        attempts = 0
        response = self.simulated_user.invoke({})
        while not (response.content.strip().upper() == "SHIFT" or response.content.strip().isdigit()) and attempts < 5:
            response = self.simulated_user.invoke({})
            attempts += 1
        
        if response.content.strip().upper() == "SHIFT":
            self._prev_data_idx = None
            self._consec_same = 0
            return {"messages": [HumanMessage(content="SHIFT")], "is_shift": True, "data_idx": self.rd_iter_idx - 1}

        data_idx = int(response.content.strip())

        # diversity guard: force SHIFT if the same data_idx has been picked too many times in a row
        if data_idx == self._prev_data_idx:
            self._consec_same += 1
        else:
            self._prev_data_idx = data_idx
            self._consec_same = 1

        if self._consec_same > self._max_consec_same:
            tqdm.write(
                f"[diversity guard] data_idx={data_idx} picked {self._consec_same} times in a row "
                f"(cap={self._max_consec_same}); forcing SHIFT."
            )
            self._prev_data_idx = None
            self._consec_same = 0
            return {"messages": [HumanMessage(content="SHIFT")], "is_shift": True, "data_idx": self.rd_iter_idx - 1}

        # This response is an AI message - we need to flip this to be a human message
        question = self.rd['question'][data_idx]
        
        # memory = state.get("memory", []) + [HumanMessage(content=question)]
        # user_memory = state.get("user_memory", []) + [HumanMessage(content=response.content)]
        return {
            "messages": [HumanMessage(content=question)],
            # "user_messages": [HumanMessage(content=response.content)],
            # "history": self.history,
            # "memory": memory,
            "is_shift": False,
            "data_idx": data_idx,
            # "user_memory": user_memory,
        }

    def should_continue(self, state):
        messages = state["messages"]
        if messages[-1].content == "SHIFT":
            return "shift"
        else:
            return "continue"
        
    def should_continue_labeler(self, state):
        if self.idx >= self.T:
            return "end"
        else:
            return "continue"




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
    
    
    history: list
    is_shift: bool
    data_idx: int
