    def restore_checkpoints(self, dir_, ignore_absence=False, restore_list=None):
        if dir_ is None:
            return
        else:
            from tensorflow.python import pywrap_tensorflow
            reader = pywrap_tensorflow.NewCheckpointReader(dir_)
            var_to_shape_map = reader.get_variable_to_shape_map()

            savedVarsName = [key.split(":")[0] for key in var_to_shape_map]

            def checker(var_name):
                if ignore_absence:
                    if var_name not in savedVarsName:
                        return False
                if restore_list is None:
                    return True
                for p in restore_list:
                    if p in var_name:
                        return True
                return False

            globalVars = [(v.name.split(":")[0], v) for v in tf.global_variables()]
            globalVarsName = [v[0] for v in globalVars]
            restoreVarsName = [v[0] for v in globalVars if checker(v[0])]
            restoreVars = [v[1] for v in globalVars if checker(v[0])]

            all_var_name_list = savedVarsName + restoreVarsName + globalVarsName
            all_var_name_list = sorted(list(set(all_var_name_list)))

            print("""
            < Legend >
            G: Defined as global variable.
            R: Includes restore variable name.
            S: Saved in checkpoint.

            Ignore absence in checkpoint: {}
            """.format("TRUE" if ignore_absence else "FALSE"))

            print("---------+-------------+")
            print("G  R  S  | Name")
            print("---------+-------------+")
            for v_name in all_var_name_list:
                G = "G" if v_name in globalVarsName else "-"
                R = "R" if v_name in restoreVarsName else "-"
                S = "S" if v_name in savedVarsName else "-"

                print("{}  {}  {}  {}".format(G, R, S, v_name))
            saver = tf.train.Saver(var_list=restoreVars)
            saver.restore(self.sess, dir_)
