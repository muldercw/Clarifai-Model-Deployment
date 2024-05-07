import os
import shutil

# Decided to just parse it manually since it's less overhead
class MyModelConfig:
    def __init__(self):
        self.name = "modelname"
        self.max_batch_size = 1
        self.input_name = "image"
        self.input_data_type = "TYPE_UINT8"
        self.input_dims = [-1, -1, -1]
        self.output_predicted_bboxes_name = "predicted_bboxes"
        self.output_predicted_bboxes_data_type = "TYPE_FP32"
        self.output_predicted_bboxes_dims = [-1, 4]
        self.output_predicted_labels_name = "predicted_labels"
        self.output_predicted_labels_data_type = "TYPE_INT32"
        self.output_predicted_labels_dims = [-1, 1]
        self.output_predicted_labels_label_filename = "labels.txt"
        self.output_predicted_scores_name = "predicted_scores"
        self.output_predicted_scores_data_type = "TYPE_FP32"
        self.output_predicted_scores_dims = [-1, 1]
        self.instance_group_count = 1
        self.instance_group_kind = "KIND_GPU"
        self.dynamic_batching_max_queue_delay_microseconds = 500
        self.parameters_key = "EXECUTION_ENV_PATH"
        self.parameters_value_string_value = "path-here"
        self.backend = "python"
    
    def __str__(self):
        return f"""\
name: "{self.name}"
max_batch_size: {self.max_batch_size}
input {{
  name: "{self.input_name}"
  data_type: {self.input_data_type}
  dims: {self.input_dims}
}}
output {{
  name: "{self.output_predicted_bboxes_name}"
  data_type: {self.output_predicted_bboxes_data_type}
  dims: {self.output_predicted_bboxes_dims}
}}
output {{
  name: "{self.output_predicted_labels_name}"
  data_type: {self.output_predicted_labels_data_type}
  dims: {self.output_predicted_labels_dims}
  label_filename: "{self.output_predicted_labels_label_filename}"
}}
output {{
  name: "{self.output_predicted_scores_name}"
  data_type: {self.output_predicted_scores_data_type}
  dims: {self.output_predicted_scores_dims}
}}
instance_group {{
  count: {self.instance_group_count}
  kind: {self.instance_group_kind}
}}
dynamic_batching {{
  max_queue_delay_microseconds: {self.dynamic_batching_max_queue_delay_microseconds}
}}
parameters {{
  key: "{self.parameters_key}"
  value {{
    string_value: "{self.parameters_value_string_value}"
  }}
}}
backend: {self.backend}
"""

def update_config(modeldirectory):
  CONFIG_FILE = f'{modeldirectory}\config.pbtxt'
  modelFolder = os.path.basename(modeldirectory)
  MODEL_DIRECTORY_PLACEHOLDER = "$$TRITON_MODEL_DIRECTORY"
  model_config = MyModelConfig()
  current_section = None
  dims_accumulator = []  
  with open(CONFIG_FILE, 'r') as f:
      for line in f:
          line = line.strip()
          if not line:
              continue
          if line.endswith('{'):
              current_section = line.split()[0]
              continue
          if line == '}':
              current_section = None
              continue
          key, value = line.split(':', 1)
          key = key.strip()
          value = value.strip()
          if key == 'max_batch_size':
              setattr(model_config, key, int(1))
          elif key == 'string_value':
              value = value.strip('"')
              if value.endswith('.gz'):
                  value = os.path.splitext(os.path.basename(value))[0]
              if value.endswith('.tar'):
                  value = value[:-4]
              value = f"{MODEL_DIRECTORY_PLACEHOLDER}/{value}.tar.gz"
              setattr(model_config, 'parameters_value_string_value', value)
          elif key == 'dims':
              if value.startswith('[') and value.endswith(']'):
                  dims_accumulator = [int(dim) for dim in value[1:-1].split(',')]
              else:
                  dims_accumulator = [int(dim) for dim in value.split(',')]
          elif current_section:
              setattr(model_config, f'{current_section}.{key}', value)
          elif key == 'name':
              setattr(model_config, key, modelFolder)
          else:
              setattr(model_config, key, value)
  if dims_accumulator:
      setattr(model_config, f'{current_section}_dims', dims_accumulator)
  #shutil.copy(CONFIG_FILE, f'{modeldirectory}\config_original.txt')
  with open(CONFIG_FILE, 'w') as f:
      f.write(str(model_config))

if __name__ == '__main__':
    update_config()