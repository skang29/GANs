    def restore_checkpoints(self, dir_, restore_list=None):
        if dir_ is None:
            return
        else:
            from tensorflow.python import pywrap_tensorflow
            reader = pywrap_tensorflow.NewCheckpointReader(dir_)
            var_to_shape_map = reader.get_variable_to_shape_map()

            savedVarsName = [key.split(":")[0] for key in var_to_shape_map]

            def checker(var_name, restore_list):
                if restore_list is None:
                    return True
                for p in restore_list:
                    if p in var_name and self.main_tower_name in var_name:
                        return True
                return False

            globalVars = [(v.name.split(":")[0], v) for v in tf.global_variables()]
            globalVarsName = [v[0] for v in globalVars]
            restoreVarsName = [v[0] for v in globalVars if checker(v[0], restore_list)]
            restoreVars = [v[1] for v in globalVars if checker(v[0], restore_list)]

            all_var_name_list = savedVarsName + restoreVarsName + globalVarsName
            all_var_name_list = sorted(list(set(all_var_name_list)))

            print("\n---------+-------------+")
            print("G  R  S  | Name")
            print("---------+-------------+")
            for v_name in all_var_name_list:
                G = "O" if v_name in globalVarsName else "-"
                R = "O" if v_name in restoreVarsName else "-"
                S = "O" if v_name in savedVarsName else "-"

                print("{}  {}  {}  {}".format(G, R, S, v_name))
            saver = tf.train.Saver(var_list=restoreVars)
            saver.restore(self.sess, dir_)
