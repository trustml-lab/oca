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
from models.dialog import DialogLearner, _get_openai_api_key

    
    
class StaticDialogLearner(DialogLearner):
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
        self.labeler = StaticLabeler(model)
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
                "end": END,
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
        
        
    def simulate(self, rd, updated_params=None):
        # self.params = updated_params if updated_params else self.params
        
        self.iterd = iter(rd)
        
        init_data = next(self.iterd)
        self.context = init_data["context"]
        self.id = init_data["id"]
        self.title = init_data["title"]
        
        chunk_log = []
        pbar = tqdm(total=500, desc="Executing states")
        
        for chunk in self.simulation.stream({"messages": []}, config={"recursion_limit": 50000}):
            
                
            if END not in chunk:
                tqdm.write(str(chunk))
                tqdm.write("---")
                
                
                chunk_log.append(chunk)
            
            if "user" in chunk and (chunk["user"]["messages"][-1].content == "NoData" or chunk["user"]["messages"][-1].content == "SHIFT"):
                tqdm.write("="*30)
                pbar.update(1)
        
        x_split_points = self.x_split#[0] + self.x_split + [self.labeler.osg.T]
        shift_split_points = self.shift
        print(x_split_points)
        
        fdata = 0
        for data in chunk_log:
            if 'labeler_osg' in data and data['labeler_osg']['label'] == 0:
                fdata += 1
        ylim = (fdata / self.labeler.x_idx) if self.labeler.x_idx > 0 else self.params.alpha
        if self.labeler.x_idx == 0:
            print("Warning: no labeling steps were recorded (labeler.x_idx == 0). Empirical FDR defaults to alpha.")
        print("Empirical FDR:", ylim)
        # print(result)
        result = []
        for idx, data in enumerate(chunk_log):
            # if 'user' in data:
            #     data['user']['messages'] = [m.content for m in data['user']['messages']]
            #     data['user']['user_messages'] = [m.content for m in data['user']['user_messages']]
            #     data['user']['memory'] = [m.content for m in data['user']['memory']]
            #     data['user']['user_memory'] = [m.content for m in data['user']['user_memory']]
            # if 'chat_bot' in data:
            #     data['chat_bot']['messages'] = [m.content for m in data['chat_bot']['messages']]
            #     data['chat_bot']['memory'] = [m.content for m in data['chat_bot']['memory']]
            if 'labeler_osg' in data:
                logprobs = chunk_log[idx-1]['chat_bot']['logprobs']
                label = data['labeler_osg']['label']
                t = data['labeler_osg']['t']
                f_std = data['labeler_osg']['score']
                question = chunk_log[idx-2]['user']['messages'][-1].content
                generation = chunk_log[idx-1]['chat_bot']['messages'][-1].content
                context = chunk_log[idx-2]['user']['context']
                samples = chunk_log[idx-1]['chat_bot']['samples']
                sample_score = chunk_log[idx-1]['chat_bot']['sample_score']
                result.append(
                    {
                        't': t,
                        'question': question,
                        'generated_sequence': generation,
                        'context': context,
                        'logprobs': logprobs,
                        'label': label,
                        'samples': samples,
                        'f_std': f_std,
                        'sample_score': sample_score,
                    }
                )
        json.dump(result, open(os.path.join(self.params.output_dir, f'chunk_{self.params.static_dialog_chunk}.json'), 'w'), indent=4)
        # json.dump(self.labeler.osg.fdrs, open(os.path.join(self.params.output_dir, 'fdrs.json'), 'w'), indent=4)
        # json.dump(self.labeler.osg.efficiencies, open(os.path.join(self.params.output_dir, 'efficiencies.json'), 'w'), indent=4)
        # json.dump(self.labeler.osg.weights.tolist(), open(os.path.join(self.params.output_dir, 'weights.json'), 'w'), indent=4)
        # json.dump(x_split_points, open(os.path.join(self.params.output_dir, 'x_split_points.json'), 'w'), indent=4)
        # json.dump(shift_split_points, open(os.path.join(self.params.output_dir, 'shift_split_points.json'), 'w'), indent=4)
        
        
    
    
    
    
    def my_chat_bot(self, messages):
        system_message = {
            "role": "system",
            "content": "You are a helpful AI assistant for question answering. Answer the question from the user in one or two sentences. \
                **Do not ask questions** — your role is to **answer**, not inquire.\n\nDo not say 'I don't know' or 'I can't answer that'.",
        }
        messages = [system_message] + messages
        want_logprobs = getattr(self.params, "confidence_score", None) == "fM1"
        completion_params = dict(
            model=self.params.model,
            messages=messages,
            seed=123,
            max_tokens=500,
            temperature=0.0,
        )
        if want_logprobs:
            completion_params["logprobs"] = True

        completion = self.client.chat.completions.create(**completion_params)

        # Some models/endpoints don't support `n`; fall back to repeated single-sample calls.
        try:
            samples = self.client.chat.completions.create(
                model=self.params.model,
                messages=messages,
                seed=123,
                max_tokens=500,
                temperature=1.0,
                n=10,  # Generate 10 samples
            )
            sample_choices = samples.choices
        except Exception:
            sample_choices = []
            for _ in range(10):
                s = self.client.chat.completions.create(
                    model=self.params.model,
                    messages=messages,
                    seed=123,
                    max_tokens=500,
                    temperature=1.0,
                )
                sample_choices.append(s.choices[0])
            
        if want_logprobs and getattr(completion.choices[0], "logprobs", None) is not None:
            logprobs = [token.logprob for token in completion.choices[0].logprobs.content]
        else:
            logprobs = []

        return completion.choices[0].message.model_dump(), logprobs, sample_choices

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
        
        sample_scores = [self.sample_score(chat_bot_response["content"], sample) for sample in [choice.message.content for choice in chat_bot_samples]]
        sample_scores = sum(sample_scores) / len(sample_scores) / 10.0
        
        memory = state.get("memory", []) + [AIMessage(content=chat_bot_response["content"])]
        # Respond with an AI Message
        return {
            "messages": [AIMessage(content=chat_bot_response["content"])],
            # "generation": chat_bot_response["content"],
            "logprobs": logprobs,
            "memory": memory,
            "samples": [choice.message.content for choice in chat_bot_samples],
            "sample_score": sample_scores,
        }

    def simulated_user_node(self, state):
        # messages = state["user_messages"]
        messages = state.get("user_memory", [])
        # messages = messages[-5:]
        if (len(messages) > 1 and messages[-1].content == "SHIFT") or len(messages) >= self.max_turn:
            state["memory"] = [] #  empty memory.
            state["user_memory"] = [] #  empty memory.
            new_messages = state.get("user_memory", [])
            try:
                init_data = next(self.iterd)
            except StopIteration:
                return {"messages": [HumanMessage(content="NoData")]}

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
        model = ChatOpenAI(model="gpt-4o", temperature=0.0, max_tokens=500)

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
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user}
            ],
            temperature=0.0,
            max_tokens=10
        )
        return int(resp.choices[0].message.content.strip())
        
    
        

    def should_continue(self, state):
        messages = state["messages"]
        if  messages[-1].content == "NoData":
            return "end"
        elif messages[-1].content == "SHIFT":
            return "shift"
        else:
            return "continue"
        
    def should_continue_labeler(self, state):
        return "continue"










class StaticLabeler:
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
    
    def __init__(self, model):
        self.max_tokens=500
        self.temperature=0
        self.stop=None
        self.seed=123
        self.tools=None
        self.logprobs=False
        self.top_logprobs=None
        # self.model = "o4-mini"
        
        self.model = "gpt-4o"
        
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
            "seed": self.seed,
            "logprobs": self.logprobs,
            "top_logprobs": self.top_logprobs,
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
                "logprobs": self.logprobs,
                "top_logprobs": self.top_logprobs,
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
        

        score = np.exp(np.array(state["logprobs"]).mean())
        # index = self.osg.predict()
        # self.osg.update(score, index, label)
        self.x_idx += 1
        
        # # print("step / T:", f"{self.osg.t} / {self.osg.T}")
        # state["messages"] = state["messages"][-1]
        result_mesage = AIMessage(content=state["messages"][-1].content)# if self.osg.tau_h[index] <= score \
                        #else AIMessage(content="I don't know.. This is out of my scope. Can you ask me something else?")
        user_memory = state.get("user_memory", []) + [result_mesage]
        # if self.osg.tau_h[index] > score:
        #     return {
        #         "user_messages": [AIMessage(content="I don't know.. This is out of my scope. Can you ask me something else?")],
        #         "user_memory": user_memory,
        #         "label": label,
        #         "IDK_flag": True,
        #         "t": self.x_idx,
        #     }
        
        return {
            "user_messages": [result_mesage],
            "user_memory": user_memory,
            "label": label,
            # "IDK_flag": False if self.osg.tau_h[index] <= score else True,
            "t": self.x_idx,
            "score": score,
            # "index": index,
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
