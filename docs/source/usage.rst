Usage
=====

Bagian 1 - Chose Your Input Image
----------------------------------

Ini merupakan bagian awal dari proses manipulasi gambar, yaitu dengan memilih terlebih dahulu gambar input yang nantinya akan diaplikasikan style dari gambar style reference:

.. code-block:: 

   """
   Bagian  : Proses Input Gambar - Tahap 1
   Revisi  : 1
   Tanggal : 2023/05/31
   """
   # Accept file from user
   uploaded_file = st.file_uploader("Your input image", type="jpg")

   # Store the uploaded image as a local file in the "test_input" folder
   if uploaded_file is None:
       # with open(os.path.join("test_input", uploaded_file.name), "wb") as f:
       #     f.write(uploaded_file.getbuffer())
       # st.write("Saved file:", uploaded_file.name)

       # Diberikan indentasi agar dapat dijalankan - Aldrin
       # Set the dpi of the figure, we change it to 100 because for performance reasons
       plt.rcParams["figure.dpi"] = 100

       # Choose input face : Add your own image to the test_input directory and put the name here
       # filepath = f"test_input/{uploaded_file.name}"
       filepath = f"test_input/iu.jpeg"

       # Strip the extension and add .pt
       name = strip_path_extension(filepath) + ".pt"

       # Aligns and crops face
       aligned_face = align_face(filepath)

       # Use e4e to encode the face (act as GAN inversion)
       my_w = e4e_projection(aligned_face, name, device).unsqueeze(0)

       # Display the input face
       st.image(filepath, caption="Input Image", width=300)

Bagian 2 - Chose Your Own Style
-------------------------------

Ini merupakan tahap kedua dari proses manipulasi gambar, yaitu dengan memilihi gambar style reference yang akan diekstrak style-nya dan diaplikasikan pada gambar input:

.. code-block::

   """
   Bagian  : Proses Pembuatan Model Style dari Gambar Reference - Tahap 2
   Revisi  : 1
   Tanggal : 2023/05/31
   """
   # Upload your own style images into the style_images folder
   # and type it into the field in the following format without
   # the directory name.
   # Caution! Upload multiple style images to do multi-shot image translation,
   # but the more images you upload, the longer it will take to run and the more memory it will consume.

   # Accept file from user
   uploaded_file = st.file_uploader("Your style reference image", type="jpg")

   # Store the uploaded image as a local file in the "style_images" folder
   if uploaded_file is None:
       # with open(os.path.join("style_images", uploaded_file.name), "wb") as f:
       #     f.write(uploaded_file.getbuffer())
       # st.write("Saved file:", uploaded_file.name)

       # Diberikan indentasi agar dapat dijalankan - Aldrin
       # names = [uploaded_file.name]
       names = ['sketch.jpeg']

       targets = []
       latents = []

       for name in names:
           style_path = os.path.join("style_images", name)
           assert os.path.exists(style_path), f"{style_path} does not exist!"

           name = strip_path_extension(name)

           # crop and align the face
           style_aligned_path = os.path.join("style_images_aligned", f"{name}.png")
           if not os.path.exists(style_aligned_path):
               style_aligned = align_face(style_path)
               style_aligned.save(style_aligned_path)
           else:
               style_aligned = Image.open(style_aligned_path).convert("RGB")

           # GAN invert
           style_code_path = os.path.join("inversion_codes", f"{name}.pt")
           if not os.path.exists(style_code_path):
               latent = e4e_projection(style_aligned, style_code_path, device)
           else:
               latent = torch.load(style_code_path)["latent"]

           targets.append(transform(style_aligned).to(device))
           latents.append(latent.to(device))

       targets = torch.stack(targets, 0)
       latents = torch.stack(latents, 0)
  
Bagian 3 - Finetune StyleGAN
----------------------------
Ini merupakan tahap ke-3 dari proses manipulasi gambar, yaitu proses di mana styleGAN akan dilakukan finetuning untuk menyesuaikan keinginan pengguna.

.. code-block::

   """
   Bagian  : Fintune StyleGAN - Tahap 3
   Revisi  : 1
   Tanggal : 2023/05/31
   """
   # alpha controls the strength of the style
   alpha = st.slider("Style Strength", 0.0, 1.0, 0.5, 0.1)
   alpha = 1 - alpha

   # Tries to preserve color of original image by limiting family of allowable transformations. Set to false if you want to transfer color from reference image. This also leads to heavier stylization
   preserve_color = st.checkbox("Preserve Color", value=False)

   # Number of finetuning steps. Different style reference may require different iterations. Try 200~500 iterations.
   # But, we limit the number of iterations to 350 for performance reasons.
   num_iter = st.number_input(
     "Number of iterations", value=200, step=1, min_value=200, max_value=350
   )

   # Log training on wandb and interval for image logging
   use_wandb = st.checkbox("Use wandb", value=False)
   log_interval = st.number_input(
     "Log interval", value=10, step=1, min_value=1, max_value=100
   )

   if use_wandb:
     wandb.init(project="JoJoGAN")
     config = wandb.config
     config.num_iter = num_iter
     config.preserve_color = preserve_color
     wandb.log(
         {"Style reference": [wandb.Image(transforms.ToPILImage()(target_im))]}, step=0
     )

   # load discriminator for perceptual loss
   discriminator = Discriminator(1024, 2).eval().to(device)
   ckpt = torch.load(
     "models/stylegan2-ffhq-config-f.pt", map_location=lambda storage, loc: storage
   )
   discriminator.load_state_dict(ckpt["d"], strict=False)

   # reset generator
   del generator
   generator = deepcopy(original_generator)

   g_optim = optim.Adam(generator.parameters(), lr=2e-3, betas=(0, 0.99))

   # Which layers to swap for generating a family of plausible real images -> fake image
   if preserve_color:
     id_swap = [9, 11, 15, 16, 17]
   else:
     id_swap = list(range(7, generator.n_latent))

   for idx in tqdm(range(num_iter)):
     mean_w = (
         generator.get_latent(torch.randn([latents.size(0), latent_dim]).to(device))
         .unsqueeze(1)
         .repeat(1, generator.n_latent, 1)
     )
     in_latent = latents.clone()
     in_latent[:, id_swap] = (
         alpha * latents[:, id_swap] + (1 - alpha) * mean_w[:, id_swap]
     )

     img = generator(in_latent, input_is_latent=True)

     with torch.no_grad():
         real_feat = discriminator(targets)
     fake_feat = discriminator(img)

     loss = sum([F.l1_loss(a, b) for a, b in zip(fake_feat, real_feat)]) / len(fake_feat)

     if use_wandb:
         wandb.log({"loss": loss}, step=idx)
         if idx % log_interval == 0:
             generator.eval()
             my_sample = generator(my_w, input_is_latent=True)
             generator.train()
             my_sample = transforms.ToPILImage()(
                 utils.make_grid(my_sample, normalize=True, range=(-1, 1))
             )
             wandb.log({"Current stylization": [wandb.Image(my_sample)]}, step=idx)

     g_optim.zero_grad()
     loss.backward()
     g_optim.step()
     
Bagian 4 - Generate Result
--------------------------
Ini merupakan tahap terakhir dari proses manipulasi gambar, yaitu dengan men-generate hasil dari pengaplikasian style kepada gambar input.

.. code-block::

   """
   Bagian  : Generate Result - Tahap 4
   Revisi  : 1
   Tanggal : 2023/05/31
   """
   n_sample =  1
   seed = 1000

   torch.manual_seed(seed)
   with torch.no_grad():
     generator.eval()
     z = torch.randn(n_sample, latent_dim, device=device)

     original_sample = original_generator([z], truncation=0.7, truncation_latent=mean_latent)
     sample = generator([z], truncation=0.7, truncation_latent=mean_latent)

     original_my_sample = original_generator(my_w, input_is_latent=True)
     my_sample = generator(my_w, input_is_latent=True)

   # display reference images
   style_images = []
   for name in names:
     style_path = f'style_images_aligned/{strip_path_extension(name)}.png'
     style_image = transform(Image.open(style_path))
     style_images.append(style_image)

   face = transform(aligned_face).to(device).unsqueeze(0)
   style_images = torch.stack(style_images, 0).to(device)
   st.image(transforms.ToPILImage()(utils.make_grid(style_images, normalize=True, range=(-1, 1))), caption="Style Reference", width=300)

   my_output = torch.cat([face, my_sample], 0)
   st.image(transforms.ToPILImage()(utils.make_grid(my_output, normalize=True, range=(-1, 1))), caption="My sample", width=300)
