
import numpy as np
import mujoco_py.modder

class TextureModder(mujoco_py.modder.TextureModder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cache_stripe_matrices()

    def brighten(self, name, amount):
        bitmap = self.get_texture(name).bitmap
        bitmap[:] = np.clip(bitmap[:].astype(np.int32) + amount, 0, 255)
        self.upload_texture(name)
        return bitmap

    def add_blotch(self, name):
        """add a varied size patch somewhere on bitmap"""
        pass

    def rand_all(self, name):
        choices = [
            self.rand_checker,
            self.rand_stripes,
            self.rand_gradient,
            self.rand_rgb,
            self.rand_noise,
        ]
        choice = self.random_state.randint(len(choices))
        return choices[choice](name)

    def get_stripe_matrices(self, name):
        o = 'h' if np.random.binomial(1,0.5) else 'v'
        if name == 'skybox':
            return self._skybox_stripe_mat[o]
        else:
            geom_id = self.model.geom_name2id(name)
            return self._geom_stripe_mats[o][geom_id]

    def rand_stripes(self, name):
        rgb1, rgb2 = self.get_rand_rgb(2)
        return self.set_stripes(name, rgb1, rgb2)

    def set_stripes(self, name, rgb1, rgb2):
        bitmap = self.get_texture(name).bitmap
        stp1, stp2 = self.get_stripe_matrices(name)

        rgb1 = np.asarray(rgb1).reshape([1, 1, -1])
        rgb2 = np.asarray(rgb2).reshape([1, 1, -1])
        bitmap[:] = rgb1 * stp1 + rgb2 * stp2

        self.upload_texture(name)
        return bitmap

    def _make_stripe_matrices(self, h, w, o):
        if o == 'v':
            re = np.r_[((w + 1) // 2) * [0, 1]]
            stp1 = np.expand_dims(np.row_stack((h * [re])), -1)[:h, :w]
        elif o == 'h':
            re = np.r_[((h + 1) // 2) * [0, 1]]
            stp1 = np.expand_dims(np.column_stack((h * [re])), -1)[:h, :w]
        stp2 = np.invert(stp1)
        return stp1, stp2

    def _cache_stripe_matrices(self):
        self._geom_stripe_mats = dict(v=[], h=[])
        for geom_id in range(self.model.ngeom):
            mat_id = self.model.geom_matid[geom_id]
            tex_id = self.model.mat_texid[mat_id]
            texture = self.textures[tex_id]
            h, w = texture.bitmap.shape[:2]
            self._geom_stripe_mats['v'].append(self._make_stripe_matrices(h, w, 'v'))
            self._geom_stripe_mats['h'].append(self._make_stripe_matrices(h, w, 'h'))

        # add skybox
        skybox_tex_id = -1
        for tex_id in range(self.model.ntex):
            skybox_textype = 2
            if self.model.tex_type[tex_id] == skybox_textype:
                skybox_tex_id = tex_id
        self._skybox_stripe_mat = {}
        if skybox_tex_id >= 0:
            texture = self.textures[skybox_tex_id]
            h, w = texture.bitmap.shape[:2]
            self._skybox_stripe_mat['v'] = self._make_stripe_matrices(h, w, 'v')
            self._skybox_stripe_mat['h'] = self._make_stripe_matrices(h, w, 'h')
        else:
            self._skybox_stripe_mat['v'] = None
            self._skybox_stripe_mat['h'] = None


