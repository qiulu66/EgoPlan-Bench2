from lmdeploy import pipeline, TurbomindEngineConfig, ChatTemplateConfig
from lmdeploy.vl import load_image
from lmdeploy.vl.constants import IMAGE_TOKEN

class VLM():
    def __init__(self, weight_dir):
        super().__init__()
        chat_template_config = ChatTemplateConfig('internvl-internlm2')
        self.pipe = pipeline(weight_dir, chat_template_config=chat_template_config, backend_config=TurbomindEngineConfig())

    def inference(self, question, images_path):
        images = [load_image(image_path) for image_path in images_path]

        prompt = ''
        for idx in range(len(images_path)):
            prompt += f'Image-{str(idx + 1)}: {IMAGE_TOKEN}\n'
        prompt += question
        
        print('\nprompt: ', prompt)
        response = self.pipe((prompt, images))
        print('\nvlm_answer: ', response.text)

        return response.text