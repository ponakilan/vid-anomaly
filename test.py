from future_frame_pred_torch import *

state_dict = torch.load("models/IPAD_R01.pth")
model = FramePredictor()
model.load_state_dict(state_dict)
print(model)