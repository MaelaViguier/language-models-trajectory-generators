import numpy as np
import matplotlib.pyplot as plt
import sys
import torch
import config
from PIL import Image
from torchvision import transforms
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks

sys.path.append("./XMem/")

from XMem.inference.inference_core import InferenceCore
from XMem.inference.interact.interactive_utils import image_to_torch, index_numpy_to_one_hot_torch, torch_prob_to_numpy_mask, overlay_davis

#from google.genai import types

# def get_langsam_output(image, model, segmentation_texts, segmentation_count):

#     segmentation_texts = " . ".join(segmentation_texts)

#     masks, boxes, phrases, logits = model.predict(image, segmentation_texts)

#     _, ax = plt.subplots(1, 1 + len(masks), figsize=(5 + (5 * len(masks)), 5))
#     [a.axis("off") for a in ax.flatten()]
#     ax[0].imshow(image)

#     for i, (mask, box, phrase) in enumerate(zip(masks, boxes, phrases)):
#         to_tensor = transforms.PILToTensor()
#         image_tensor = to_tensor(image)
#         box = box.unsqueeze(dim=0)
#         image_tensor = draw_bounding_boxes(image_tensor, box, colors=["red"], width=3)
#         image_tensor = draw_segmentation_masks(image_tensor, mask, alpha=0.5, colors=["cyan"])
#         to_pil_image = transforms.ToPILImage()
#         image_pil = to_pil_image(image_tensor)

#         ax[1 + i].imshow(image_pil)
#         ax[1 + i].text(box[0][0], box[0][1] - 15, phrase, color="red", bbox={"facecolor":"white", "edgecolor":"red", "boxstyle":"square"})

#     plt.savefig(config.langsam_image_path.format(object=segmentation_count))
#     plt.show()

#     masks = masks.float()

#     return masks, boxes, phrases

def get_langsam_output(image, model, segmentation_texts, segmentation_count):

    if isinstance(segmentation_texts, str):  
        segmentation_texts = [segmentation_texts]

    [out] = model.predict([image], segmentation_texts)
    print(out,len(out))
    masks, boxes, phrases = out['masks'],out['boxes'],out['labels']
    # masks = masks.astype(bool)
    masks, boxes, phrases =np.array([masks[0]]), np.array([boxes[0]]), np.array([phrases[0]])
    masks = torch.tensor(masks, dtype=torch.bool)
    print(masks)
    print(masks.shape)
    _, ax = plt.subplots(1, 1 + len(masks), figsize=(5 + (5 * len(masks)), 5))
    [a.axis("off") for a in ax.flatten()]
    ax[0].imshow(image)
    to_be_discarded = []
    for i, (mask, box, phrase) in enumerate(zip(masks, boxes, phrases)):
        to_tensor = transforms.PILToTensor()
        image_tensor = to_tensor(image)
        if box[1]>image.size[1]-30:
            to_be_discarded.append(i)
            continue
        box = torch.tensor(box)
        box = box.unsqueeze(dim=0)
        image_tensor = draw_bounding_boxes(image_tensor, box, colors=["red"], width=3)
        image_tensor = draw_segmentation_masks(image_tensor, mask, alpha=0.5, colors=["cyan"])
        to_pil_image = transforms.ToPILImage()
        image_pil = to_pil_image(image_tensor)

        ax[1 + i].imshow(image_pil)
        ax[1 + i].text(box[0][0], box[0][1] - 15, phrase, color="red", bbox={"facecolor":'white', "edgecolor":"red", "boxstyle":"square"})

    plt.savefig(config.langsam_image_path.format(object=segmentation_count))
    plt.show()
    

    masks = masks.float()
    for tbd in to_be_discarded:
        boxes.pop(tbd)
    return masks, boxes, phrases


def get_chatgpt_output(client, model, new_prompt, messages, role, file=sys.stdout):

#chatgpt
    # print(role + ":", file=file)
    # print(new_prompt, file=file)
    # messages.append({"role":role, "content":new_prompt})

    # completion = client.chat.completions.create(
    #     model=model,
    #     temperature=0,
    #     messages=messages,
    #     stream=True
    # )

    # print("assistant:", file=file)

    # new_output = ""

    # for chunk in completion:
    #     chunk_content = chunk.choices[0].delta.content
    #     finish_reason = chunk.choices[0].finish_reason
    #     if chunk_content is not None:
    #         print(chunk_content, end="", file=file)
    #         new_output += chunk_content
    #     else:
    #         print("finish_reason:", finish_reason, file=file)

    # messages.append({"role":"assistant", "content":new_output})

#llama
    #######
    print(role + ":", file=file)
    print(new_prompt, file=file)
    messages.append({"role":role, "content":new_prompt})

    stream = client.chat(
    #completion = client.chat.completions.create(
         model=model,
         messages=messages,
         stream=True,
         #options={"tempature": 0}
         options={"tempature": 0, "num_predict":512}
         #temperature=0
    )

    print("assistant:", file=file)

    new_output = ""

    for chunk in stream:
        
        chunk_content = chunk.get("message", {}).get('content')
        #chunk_content = chunk.choices[0].delta.content
        finish_reason = chunk.get('finish_reason')
        #finish_reason = chunk.choices[0].finish_reason
        
        if chunk_content is not None :
            print(chunk_content, end="", file=file)
            new_output += chunk_content
            
        else :
            print("finish_reason:", finish_reason, file=file)
            
    messages.append({"role":"assistant", "content":new_output})

#qwen
    ######

    #print(role + ":", file=file)
    #print(new_prompt, file=file)

    #messages.append({"role": role, "content": new_prompt})

    #tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    #model = AutoModelForCausalLM.from_pretrained(model, trust_remote_code=True)
    #model = model.to("cuda" if torch.cuda.is_available() else "cpu")

    #text = tokenizer.apply_chat_template(
        #messages,
        #tokenize = False,
        #add_generation_prompt = True
    #)

    #inputs = tokenizer(text, return_tensors = 'pt').to(model.device)

    #response_ids = model.generate(**inputs, max_new_tokens=32768)[0][len(inputs.input_ids[0]):].tolist()
    #response = tokenizer.decode(response_ids, skip_special_tokens = True)

    #if file:
        #print("assistant: ", file=file)
        #print(response, file=file)

    #messages.append({"role":"assistant", "content":response})

#gemini
    ##################################
    #print(role + ":", file=file)
    #print(new_prompt, file=file)

    #messages.append({"role": role, "content": new_prompt})

    #gemini_contents = []
    #for msg in messages:
        #gemini_role = "user" if msg["role"] == "user" else "model"
        #gemini_contents.append({"role": gemini_role, "parts": [{"text": msg["content"]}]})

    #print("assistant:", file=file)

    #new_output = ""

    #try:
        #response = client.models.generate_content( 
            #model=model,
            #contents=gemini_contents, 
            #config=types.GenerateContentConfig(temperature=0), # You can add other params here 
        #)

        ##for chunk in response:
            ##chunk_content = chunk.text 

            ##if chunk_content is not None:
                ##print(chunk_content, end="", file=file)
                ##new_output += chunk_content

        #new_output=response.text
        #print(new_output, end="", file=file)


    #except Exception as e:
        #print(f"\nErreur lors de l'appel à l'API Gemini : {e}", file=file)
        #new_output = f"Error: {e}" 

    #messages.append({"role": role, "content": new_prompt})

    return messages



def get_xmem_output(model, device, trajectory_length):

    mask = np.array(Image.open(config.xmem_input_path).convert("L"))
    mask = np.unique(mask, return_inverse=True)[1].reshape(mask.shape)
    num_objects = len(np.unique(mask)) - 1

    torch.cuda.empty_cache()

    processor = InferenceCore(model, config.xmem_config)
    processor.set_all_labels(range(1, num_objects + 1))

    masks = []

    with torch.cuda.amp.autocast(enabled=True):

        for i in range(0, trajectory_length + 1, config.xmem_output_every):

            frame = np.array(Image.open(config.rgb_image_trajectory_path.format(step=i)).convert("RGB"))

            frame_torch, _ = image_to_torch(frame, device)
            if i == 0:
                mask_torch = index_numpy_to_one_hot_torch(mask, num_objects + 1).to(device)
                prediction = processor.step(frame_torch, mask_torch[1:])
            else:
                prediction = processor.step(frame_torch)

            prediction = torch_prob_to_numpy_mask(prediction)
            masks.append(prediction)

            if i % config.xmem_visualise_every == 0:
                visualisation = overlay_davis(frame, prediction)
                output = Image.fromarray(visualisation)
                output.save(config.xmem_output_path.format(step=i))

    return masks
