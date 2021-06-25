import keras


def build_model(backbone_name, backbone_source, max_len):
    """
    Build keras model
    Input: input_word_ids
    Output: predicted class label
    """
    bert_layer = get_backbone(backbone_name, backbone_source, framework='tensorflow')
    
    # word ids encoded by tokenizer
    input_word_ids = keras.Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    
    bert_output = bert_layer(input_word_ids, return_dict=True)
    
    # pooled_output is the embedding of CLS token -> linear layer -> tanh
    pooler_output = bert_output["pooler_output"]      # [batch_size, 768].
    # pooled_output represents each input sequence (sentence) as a whole
    
    # This is the output of the last layer of the bert model
    # sequence_output represents each input token in context
    sequence_output = bert_output['last_hidden_state']
    
    """
    [CLS] is the first token of every input sequence. 
    The final hidden state of this token can be used as a sentence representation 
    for classification tasks
    """
    embedding_layer = keras.layers.Lambda(lambda sequence_output: sequence_output[:, 0, :], name='embedding_layer')
    
    # Outputs [batch_size, 768] sized embeddings for each sentence
    cls_token = embedding_layer(sequence_output)
    
    
    # Add Dropout to cls embedding ?
    cls_drop = keras.layers.Dropout(0.5)(cls_token)
    
    # --- Add Custom Layers Here ---
    label = keras.Input(shape = (), name = 'label')
    out = keras.layers.Softmax(dtype='float32')(label)
    # ------------------------------
    
    # Make the Model
    model = keras.Model(inputs=[input_word_ids, label], outputs=[out])
    return model

model = build_model(hp.backbone_name, hp.backbone_source, hp.max_len)
model.summary()


def compile_model(**compile_args): 
    
    
    return model


