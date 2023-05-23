Usage
=====

.. _installation:

Installation
------------

To use Lumache, first install it using pip:

.. code-block:: console

   (.venv) $ pip install lumache

Creating recipes
----------------

To retrieve a list of random ingredients,
you can use the ``lumache.get_random_ingredients()`` function:

.. autofunction:: lumache.get_random_ingredients

The ``kind`` parameter should be either ``"meat"``, ``"fish"``,
or ``"veggies"``. Otherwise, :py:func:`lumache.get_random_ingredients`
will raise an exception.

.. autoexception:: lumache.InvalidKindError

For example:

>>> import lumache
>>> lumache.get_random_ingredients()
['shells', 'gorgonzola', 'parsley']

Bagian 1 - Chose Your Input Image
----------------------------------

Ini merupakan bagian awal dari proses manipulasi gambar, yaitu dengan memilih terlebih dahulu gambar input yang nantinya akan diaplikasikan style dari gambar style reference:

.. 
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
