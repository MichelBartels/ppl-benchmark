from mnist import mnist_bar, show_sample
if False:
    from numpyro_vae import init, prepare_batch
else:
    from pyro_vae import init, prepare_batch

batch_size = 256
plot_steps = 100_000
num_eval_samples = 4

step, generate, reconstruct = init(batch_size)
update_loss, mnist = mnist_bar(batch_size)

for i, batch in enumerate(mnist()):
    x = prepare_batch(batch)
    loss = step(x)
    update_loss(loss)
    if (i + 1) % plot_steps == 0:
        random_samples = generate(num_eval_samples)
        show_sample(random_samples, 2, 2, filename=f"samples/random_sample_{i}.png")
        reconstructed_samples = reconstruct(x[:num_eval_samples])
        show_sample(reconstructed_samples, 2, 2, filename=f"samples/reconstructed_sample_{i}.png")
