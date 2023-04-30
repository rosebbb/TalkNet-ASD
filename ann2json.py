import json
import random
import string


class JsonFormatter():
    def __init__(self, file_name, number_person, frame_count, duration, device='cuda'):
        self.file_name = file_name
        self.data = self.create_structure(file_name, number_person, frame_count, duration)
        
    def random_string(self):
        return ''.join(random.choices(string.ascii_letters, k=10))
                        #  {
                        #     "frame":1,
                        #     "enabled":true,
                        #     "rotation":0,
                        #     "x":30.22452504317789,
                        #     "y":2.715409710228363,
                        #     "width":21.243523316062177,
                        #     "height":38.68739205526771,
                        #     "time":0.04
                        #  },

    def add_seq_element(self, person_index, frame_no, x, y, width, height):
        time = frame_no * 0.04 #?
        seq_entry = {
            "frame":frame_no,
            "enabled":False,
            "rotation":0, # fixed
            "x":x,
            "y":y, 
            "width":width,
            "height":height,
            "time":time
        }
        self.data['annotations'][0]['result'][person_index]['value']['sequence'].append(seq_entry)
        return seq_entry

    def create_result_element(self, frames_count, id, duration, i):
        result_entry = {
            'value': {},
            "id": id,
            "from_name":"box",
            "to_name":"video",
            "type":"videorectangle",
            "origin":"manual",
        }

        result_entry['value'] = {
            "framesCount":frames_count, # same for all result entry
            "duration":duration, # frameCount * fps # same for all result entry
            "sequence":[],
            "labels":[str(i)],# person id in string format ["1"]
        }

        return result_entry

    def create_structure(self, file_name, number_person, frame_count, duration):
        data = [{
            'id': 3, # random
            'annotations': [],
            'file_upload': file_name,
            'drafts': [], # random
            'predictions': [], # random
            'data': {"video": "\/data\/upload\/4\/"+file_name}, 
            'meta': {}, # random
            'created_at': "2023-04-05T18:36:51.557952Z", # random
            'updated_at': "2023-04-05T20:03:30.873676Z", # random
            'inner_id': 3, # random
            'total_annotations': 1, # random
            'cancelled_annotations': 0, # random
            'total_predictions': 0, # random
            'comment_count': 0, # random
            'unresolved_comment_count': 0, # random
            'last_comment_updated_at': None, # null
            'project': 4, # random
            'updated_by': 1, # random
            'id': 0, # random
            'comment_authors': [], # random
        }]

        data['annotations'] = [{
            'id': 13, # random
            'completed_by': 1, # random
            'result': [], 
            'was_cancelled':False, # random
            'ground_truth':False, # random
            'created_at':'2023-04-05T18:56:54.985829Z', # random
            'updated_at':'2023-04-05T20:03:30.759982Z', # random
            'lead_time':5197.009, # random
            'prediction':{},
            'result_count':0, # random
            'unique_id':'6be7c022-e8f0-41cb-a810-dfc28f05b2f6', # ?
            'last_action':None,
            'task':3, # ?
            'project':4, # ?
            'updated_by':1,
            'parent_prediction':None,
            'parent_annotation':None,
            'last_created_by':None,
        }]

        for i in range(number_person):
            id = self.random_string()
            data['annotations'][0]['result'].append(self.create_result_element(frame_count, id, duration, i))
        return data



# for i in range(5):
#     print(i)
#     labels = ["i"]
#     data['annotations'][0]['result'].append(create_result_element(5909, 236.330646, labels))# 5 element



# json_data = [data]
# with open("sample.json", "w") as outfile:
#     json.dump(json_data, outfile)