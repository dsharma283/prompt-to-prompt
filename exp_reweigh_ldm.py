#!/usr/bin/env python
# coding: utf-8

# ## Copyright 2022 Google LLC. Double-click for license information.

# In[1]:


# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from exp_word_swap_ldm import *

# ## Attention Re-Weighting

output_path = f'output/reweight'

paper_fig26 = f'{output_path}/paper-fig26'
my_own = f'{output_path}/myown'

if os.path.exists(paper_fig26) is False:
    os.makedirs(paper_fig26)
if os.path.exists(my_own) is False:
    os.makedirs(my_own)

prompts = ["A photo of a tree branch at blossom"]
images, x_t = generate_source_img(save_path=None, prompts=prompts, name_hint=paper_fig26,
                                  show_ca=False, show_sa=False, seed=888)

prompts = ["A photo of a tree branch at blossom"] * 4
name_hnt = os.path.join(paper_fig26, f"tree-blossom")
equalizer = get_equalizer(prompts[0], word_select=("blossom",), values=(.5, .0, -.5))
controller = AttentionReweight(prompts, NUM_DIFFUSION_STEPS, cross_replace_steps=1., self_replace_steps=.2, equalizer=equalizer)
_ = run_and_display(prompts, controller, latent=x_t, run_baseline=False, name_hint=name_hnt)

prompts = ["A landscape with a snowy mountain"] * 4
name_hnt = os.path.join(paper_fig26, f"snowy-mountain")
equalizer = get_equalizer(prompts[0], word_select=("snowy",), values=(2.0, .0, -.5))
controller = AttentionReweight(prompts, NUM_DIFFUSION_STEPS, cross_replace_steps=1., self_replace_steps=.2, equalizer=equalizer)
_ = run_and_display(prompts, controller, latent=x_t, run_baseline=False, name_hint=name_hnt)

prompts = ["A photo of the ancient city"] * 4
name_hnt = os.path.join(paper_fig26, f"ancient-city")
equalizer = get_equalizer(prompts[0], word_select=("city",), values=(0.5, .0, -.5))
controller = AttentionReweight(prompts, NUM_DIFFUSION_STEPS, cross_replace_steps=1., self_replace_steps=.2, equalizer=equalizer)
_ = run_and_display(prompts, controller, latent=x_t, run_baseline=False, name_hint=name_hnt)

prompts = ["A crashed car"] * 4
name_hnt = os.path.join(paper_fig26, f"crashed-car")
equalizer = get_equalizer(prompts[0], word_select=("car",), values=(0.5, .0, -.5))
controller = AttentionReweight(prompts, NUM_DIFFUSION_STEPS, cross_replace_steps=1., self_replace_steps=.2, equalizer=equalizer)
_ = run_and_display(prompts, controller, latent=x_t, run_baseline=False, name_hint=name_hnt)

prompts = ["My puffy shirt"] * 4
name_hnt = os.path.join(paper_fig26, f"puffy-shirt")
equalizer = get_equalizer(prompts[0], word_select=("shirt",), values=(2.0, .0, -.5))
controller = AttentionReweight(prompts, NUM_DIFFUSION_STEPS, cross_replace_steps=1., self_replace_steps=.2, equalizer=equalizer)
_ = run_and_display(prompts, controller, latent=x_t, run_baseline=False, name_hint=name_hnt)

prompts = ["A photo of poppy field at night"] * 4
name_hnt = os.path.join(paper_fig26, f"poppy-field")
equalizer = get_equalizer(prompts[0], word_select=("night",), values=(2.0, .0, -.5))
controller = AttentionReweight(prompts, NUM_DIFFUSION_STEPS, cross_replace_steps=1., self_replace_steps=.2, equalizer=equalizer)
_ = run_and_display(prompts, controller, latent=x_t, run_baseline=False, name_hint=name_hnt)

prompts = ["A photo of beautiful peacock"] * 4
name_hnt = os.path.join(my_own, f"beautiful-peacock")
equalizer = get_equalizer(prompts[0], word_select=("beautiful",), values=(2.0, .0, -.5))
controller = AttentionReweight(prompts, NUM_DIFFUSION_STEPS, cross_replace_steps=1., self_replace_steps=.2, equalizer=equalizer)
_ = run_and_display(prompts, controller, latent=x_t, run_baseline=False, name_hint=name_hnt)

# In[20]:


#prompts = ["A photo of a poppy field at night"] * 4
#name_hnt = f"poppy-night"
#equalizer = get_equalizer(prompts[0], word_select=("night",), values=(.5, 0,  -.5))
#controller = AttentionReweight(prompts, NUM_DIFFUSION_STEPS, cross_replace_steps=1., self_replace_steps=.2, equalizer=equalizer)
#_ = run_and_display(prompts, controller, latent=x_t, run_baseline=False, name_hint=name_hnt)


# ### Edit Composition
# It might be useful to use Attention Re-Weighting with a previous edit method.

# In[21]:


#prompts = ["cake",
#           "birthday cake"] 

#name_hnt = f"bday-cake"
#lb = LocalBlend(prompts, ("cake", ("birthday", "cake")))
#controller = AttentionRefine(prompts, NUM_DIFFUSION_STEPS, cross_replace_steps=.8, self_replace_steps=.4, local_blend=lb)
#_ = run_and_display(prompts, controller, latent=x_t, run_baseline=False, name_hint=name_hnt)


#def perform_refinement_
#if __name__ == "__main__":
#    print('success')
