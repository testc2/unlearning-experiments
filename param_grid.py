from sklearn.model_selection import ParameterGrid; d = [{
    '1sampling_type':['uniform_random','targeted_random'],
    '2sampler_seed':[0,1],
    '3noise_seed':range(6),
    '4noise':[1]
},
{
    '1sampling_type':['uniform_informed','targeted_informed'],
    '2sampler_seed':[0],
    '3noise_seed':range(6),
    '4noise':[1]
}];
[print('|'.join([str(v) for v in g.values()])) for i,g in enumerate(ParameterGrid(d))]
