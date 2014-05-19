"""write meeee"""
import optimus
from ejhumphrey.chordrec.transformers import sample
from ejhumphrey.chordrec.transformers import pitch_shift
from ejhumphrey.chordrec.transformers import map_to_chroma

CHORD_FILE = "datasets/chord_labels_train0.hdf5"

# 1.1 Create Inputs
input_data = optimus.Input(
    name='cqt',
    shape=(None, 1, 5, 252))

chroma_template = optimus.Input(
    name='chroma',
    shape=(None, 12))

learning_rate = optimus.Input(
    name='learning_rate',
    shape=None)

# 1.2 Create Nodes
layer0 = optimus.Conv3D(
    name='layer0',
    input_shape=input_data.shape,
    weight_shape=(12, 1, 3, 19),
    pool_shape=(1, 3),
    act_type='relu')

layer1 = optimus.Conv3D(
    name='layer1',
    input_shape=layer0.output.shape,
    weight_shape=(16, None, 3, 15),
    pool_shape=(1, 1),
    act_type='relu')

layer2 = optimus.Affine(
    name='layer2',
    input_shape=layer1.output.shape,
    output_shape=(None, 12,),
    act_type='sigmoid')

all_nodes = [layer0, layer1, layer2]

# 1.1 Create Losses
chroma_mse = optimus.MeanSquaredError(
    name="chroma_mse")

# 2. Define Edges
trainer_edges = optimus.ConnectionManager([
    (input_data, layer0.input),
    (layer0.output, layer1.input),
    (layer1.output, layer2.input),
    (layer2.output, chroma_mse.prediction),
    (chroma_template, chroma_mse.target)])

update_manager = optimus.ConnectionManager([
    (learning_rate, layer0.weights),
    (learning_rate, layer0.bias),
    (learning_rate, layer1.weights),
    (learning_rate, layer1.bias),
    (learning_rate, layer2.weights),
    (learning_rate, layer2.bias)])

trainer = optimus.Graph(
    name='chroma_transform',
    inputs=[input_data, chroma_template, learning_rate],
    nodes=all_nodes,
    connections=trainer_edges.connections,
    outputs=[optimus.Graph.TOTAL_LOSS],
    losses=[chroma_mse],
    updates=update_manager.connections)

optimus.random_init(layer2.weights)

predictor_edges = optimus.ConnectionManager([
    (input_data, layer0.input),
    (layer0.output, layer1.input),
    (layer1.output, layer2.input)])

predictor = optimus.Graph(
    name='chroma_transform',
    inputs=[input_data],
    nodes=all_nodes,
    connections=predictor_edges.connections,
    outputs=[layer2.output])

# # 3. Create Data
source = optimus.Queue(
    optimus.File(CHORD_FILE),
    batch_size=50,
    refresh_prob=0.0,
    cache_size=500,
    transformers=[
        sample(input_data.shape[2]),
        pitch_shift(max_pitch_shift=12, bins_per_pitch=3),
        map_to_chroma])

hyperparams = {learning_rate.name: 0.02}

driver = optimus.Driver(
    graph=trainer,
    name='split0',
    output_directory='dexp/chordrec')
driver.fit(source, hyperparams=hyperparams, max_iter=25000, print_freq=250)

optimus.save(
    predictor,
    def_file='%s/chroma_transform.json' % driver.output_directory)
