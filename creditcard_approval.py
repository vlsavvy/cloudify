
# coding: utf-8

# In[2]:


from azureml import Workspace
ws = Workspace()
experiment = ws.experiments['66648eb255e7464db274700046d4b942.f-id.79f5f22e86fd40318e33f19cf52c2be2']
ds = experiment.get_intermediate_dataset(
    node_id='7ec82910-1d6d-41d0-94f6-714010ab27e9-33388',
    port_name='Results dataset',
    data_type_id='GenericCSV'
)
frame = ds.to_dataframe()


# In[3]:


frame

