def MakeAttentionScore(dataframe, data, model, fp_size, thresholds=0.5, label="true_pos"):
    global attention_score
    
    attn_ = tf.keras.Model(inputs = model.inputs, outputs = model.layers[2].output) # attention layer
    
    attention_score = attn_.predict(data)
    attention_score[:, :fp_size] = attention_score[:, :fp_size] * data.iloc[:, :fp_size]
    attention_score = pd.DataFrame(attention_score)
    
    
    if label == "true_pos":
        pred = model.predict(data)
        pred[pred >= thresholds] = 1
        pred[pred < thresholds] = 0
        pred = pd.DataFrame(pred)
        pred_idx = pred[pred[0]==1].index
        dataframe = dataframe.loc[pred_idx]
        true_positive_idx = dataframe[dataframe['toxicity']==1].index    
        attention_score = attention_score.loc[true_positive_idx]
        attention_score = attention_score.transpose()
        
    elif label == "pred_pos":
        pred = model.predict(data)
        pred[pred >= thresholds] = 1
        pred[pred < thresholds] = 0
        pred = pd.DataFrame(pred)
        pred_idx = pred[pred[0]==1].index
        
        attention_score = attention_score.loc[pred_idx]
        attention_score = attention_score.transpose()
        
    elif label == "all":
        attention_score = attention_score.transpose()
        
    return attention_score



def MakeAttentionIndex(attention_score, dataframe):
    attention_index = attention_score.transpose().index
    inform = dataframe.loc[attention_index]
    return inform

def FindDrug(inform, drug_name):
    if type(drug_name) == str:
        selected_drug = inform[inform['name']==drug_name]
        print('INDEX')
        print(selected_drug)
    else:
        raise ValueError('The \'drug_name\' value must be string type')



def AttentionAnalysis(mols=None, attention_score=None, dataframe=None, bit_info_list=None, index=None, drug_name=None, rank=10, fp_size=None):
    highlightAtomLists = [] # molecular substructure atom for highlights
    rank_information = []
    legends = [] # molecular substructure
    
    # Check the Input Values
    if mols is None:
        raise ValueError('Please input the \'mols\' value, it must be list of mol.')
        
    if attention_score is None:
        raise ValueError('Please input the \'attention_score\' value, it must be dataframe.')
        
    if dataframe is None:
        raise ValueError('Please input the \'dataframe\' value, it must be dataframe.')
    
    if bit_info_list is None:
        raise ValueError('Please input the \'bit_info_list\' value, it must be list of bit_info.')
        
    if rank <= 0:
        raise ValueError('The \'rank\' value must be larger than 0, please start with 1.')
        
    if fp_size is None:
        raise ValueError('Please input the \'fp_size\' value, it must be integer.\nThe vector size is structural fingerprint size')
        
    if (index is None) and (drug_name is None):
        raise ('Both values \'index\' and \'drug_name\' are None, please input the only one of the index value or drug_name value.')
    elif (index is not None) and (drug_name is not None):
        raise ('Both values \'index\' and \'drug_name\' were entered, please input the only one of the index value or drug_name value.')
        
    # Molecule Drawing with Highlighted Hit Sub-structure
    if index != None:
        if type(index) == int:
            selected_score = attention_score.loc[:, index]
            selected_score = selected_score.sort_values(ascending=False)
            for rank_i in range(rank):
                if selected_score.index[rank_i] < fp_size: # if index < fp_size, then molecular substructure
                    selected_bit = selected_score.index[rank_i]
                    if bit_info_list[index][selected_bit][0][1] == 0: 
                        check_bits = 0
                        while bit_info_list[index][selected_bit][check_bits][1] == 0:                            
                            check_bits += 1
                            if check_bits == len(bit_info_list[index][selected_bit]):
                                selected_env = Chem.FindAtomEnvironmentOfRadiusN(mols[index], bit_info_list[index][selected_bit][check_bits-1][1], bit_info_list[index][selected_bit][check_bits-1][0])
                                break
                            elif bit_info_list[index][selected_bit][check_bits][1] != 0:
                                selected_env = Chem.FindAtomEnvironmentOfRadiusN(mols[index], bit_info_list[index][selected_bit][check_bits][1], bit_info_list[index][selected_bit][check_bits][0])
                                break                            
                    else:
                        selected_env = Chem.FindAtomEnvironmentOfRadiusN(mols[index], bit_info_list[index][selected_bit][0][1], bit_info_list[index][selected_bit][0][0])                    
                    selected_submol = Chem.PathToSubmol(mols[index], selected_env)
                    highlightAtom = mols[index].GetSubstructMatch(selected_submol)
                    highlightAtomLists.append(highlightAtom)
                    selected_SMILES = Chem.MolToSmiles(selected_submol)
                    selected_legend = f'Rank: {rank_i+1}, bit: {selected_bit}'
                    rank_information.append(f'Rank: {rank_i+1}\nAttention score: {selected_score.iloc[rank_i]:.5f}\nbit: {selected_bit}\nbit_info: {bit_info_list[index][selected_bit]}\nSMILES: {selected_SMILES}')
                    legends.append(str(selected_legend))
                elif selected_score.index[rank_i] >= fp_size: # if index >= fp_size, then chemical feature
                    selected_bit = selected_score.index[rank_i]
                    selected_legend = f'Rank: {rank_i+1}, bit: {selected_bit}'
                    rank_information.append(f'Rank: {rank_i+1}\nAttention score: {selected_score.iloc[rank_i]:.5f} \nbit: {selected_bit}')
                    legends.append(str(selected_legend))
                    highlightAtomLists.append('')  
            IPythonConsole.drawOptions.useBWAtomPalette()
            Draw_attention_score = Chem.Draw.MolsToGridImage([mols[index] for i in range(len(highlightAtomLists))],
                                                             molsPerRow=2,
                                                             subImgSize=(400,400),
                                                             highlightAtomLists=highlightAtomLists,
                                                             legends=legends,
                                                             returnPNG=False)
            for i in range(len(rank_information)):
                print(rank_information[i])
            return Draw_attention_score        
        elif type(index) != int:
            raise ValueError('The \'index\' value must be integer type')
            
    elif drug_name != None:
        if type(drug_name) == str:
            selected_drug = dataframe[dataframe['name']==drug_name]
            index = selected_drug.index
            selected_score = attention_score.loc[:, index]
            selected_score = selected_score.sort_values(ascending=False)
            for rank_i in range(rank):
                if selected_score.index[rank_i] < fp_size: # if index < fp_size, then molecular substructure
                    selected_bit = selected_score.index[rank_i]
                    if bit_info_list[index][selected_bit][0][1] == 0: 
                        check_bits = 0
                        while bit_info_list[index][selected_bit][check_bits][1] == 0:
                            if bit_info_list[index][selected_bit][check_bits][1] != 0:
                                selected_env = Chem.FindAtomEnvironmentOfRadiusN(mols[index], bit_info_list[index][selected_bit][check_bits][1], bit_info_list[index][selected_bit][check_bits][0])
                                break
                            check_bits += 1
                            if check_bits == len(bit_info_list[index][selected_bit]):
                                selected_env = Chem.FindAtomEnvironmentOfRadiusN(mols[index], bit_info_list[index][selected_bit][check_bits-1][1], bit_info_list[index][selected_bit][check_bits-1][0])
                                break
                    else:
                        selected_env = Chem.FindAtomEnvironmentOfRadiusN(mols[index], bit_info_list[index][selected_bit][0][1], bit_info_list[index][selected_bit][0][0])                    
                    selected_submol = Chem.PathToSubmol(mols[index], selected_env)
                    highlightAtom = mols[index].GetSubstructMatch(selected_submol)
                    highlightAtomLists.append(highlightAtom)
                    selected_SMILES = Chem.MolToSmiles(selected_submol)
                    selected_legend = f'Rank: {rank_i+1}, bit: {selected_bit}'
                    rank_information.append(f'Rank: {rank_i+1}\nAttention score: {selected_score.iloc[rank_i]:.5f}\nbit: {selected_bit}\nbit_info: {bit_info_list[index][selected_bit]}\nSMILES: {selected_SMILES}')
                    legends.append(str(selected_legend))
                elif selected_score.index[rank_i] >= fp_size: # if index >= fp_size, then chemical feature
                    selected_bit = selected_score.index[rank_i]
                    selected_legend = f'Rank: {rank_i+1}, bit: {selected_bit}'
                    rank_information.append(f'Rank: {rank_i+1}\nAttention score: {selected_score.iloc[rank_i]:.5f} \nbit: {selected_bit}')
                    legends.append(str(selceted_legend))
                    highlightAtomLists.append('')
            Draw_attention_score = Chem.Draw.MolsToGridImage([mols[index] for i in range(len(highlightAtomLists))],
                                                             molsPerRow=2,
                                                             subImgSize=(400,400),
                                                             highlightAtomLists=highlightAtomLists,
                                                             legends=legends,
                                                             returnPNG=False)
            for i in range(len(rank_information)):
                rank_information[i]
            return Draw_attention_score
        elif type(drug_name) != str:
            raise ValueError('The \'drug_name\' value must be string type')



