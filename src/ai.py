import os, json, requests


class TextMessage:
    
    ROLE_USER = "user"
    ROLE_AI = "assistant"
    ROLE_SYSTEM = "system"
    
    def __init__(self, sender, message):
        self.sender = sender
        self.message = message
    
    def get_data(self):
        return {
            "sender": sender,
            "message": message,
        }


class Context:
    
    def __init__(self):
        self.items = []
    
    def add_message(self, message):
        self.items.append(message)
    
    
    def get_data(self):
        return [item.get_data() for item in self.items]
    

class Provider:
    
    def get_url(self):
        return ""
    
    def get_model_name(self):
        return ""
    
    def get_api_key(self):
        return ""
    
    async def send(self, context):
        
        data = {
            "model": self.get_model_name(),
            "message": context.get_data(),
        }
        
        headers = {
            "Authorization": "Bearer " + self.get_api_key()
        }
        
        response = requests.post(
            provider.get_url(),
            headers=headers, data=json.dumps(data)
        )
        
        response.raise_for_status()
        result = response.json()
        
        return result


class OpenRouterProvider(Provder):
    
    def __init__(self, api_key="", model_name="", api_key=""):
        super()..__init__(self)
        self.api_key = api_key
        self.model_name = model_name
        self.api_key = api_key
    
    def getUrl(self):
        return "https://openrouter.ai/v1/cjat/completions"


class ToolRegistry:
    
    def __init__(self):
        self.items = []


class Tool:
    
    def __init__(self):
        self.name = ""
        self.description = ""
        self.schema = {}
    
    def get_schema(self):
        return self.schema
        
    async def execute(self, params):
        return None


class Agent:
    
    def __init__(self, provider, tools):
        self.provider = provider
        self.tools = tools
        
    
    async send(self, context):
        
        result = await self.provider.send(context)
        context.add_message(Context.ROLE_AI, result["choices"][0]["message"]["content"])
        
        
    async send_with_fallback(self, context):
        
        count = 0
        while count < self.max_iters:
            
            await self.send(context)
            count += 1
    
    