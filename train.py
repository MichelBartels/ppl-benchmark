from mnist import mnist_bar
if False:
    from numpyro_vae import step_fn, prepare_batch
else:
    from pyro_vae import step_fn, prepare_batch

batch_size = 256

step = step_fn(batch_size)
update_loss, mnist = mnist_bar(batch_size)

for batch in mnist():
    x = prepare_batch(batch)
    loss = step(x)
    update_loss(loss)
