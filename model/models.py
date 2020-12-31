
def create_model(opt, labeled_dataset=None, unlabeled_dataset=None):
    print(opt.model)
    if opt.model == 'wsupervised':
        from .T2model import T2NetModel
        model = T2NetModel()
    elif opt.model == 'supervised':
        from .TaskModel import TNetModel
        model = TNetModel()
    elif opt.model == 'test':
        from .test_model import TestModel
        model = TestModel()
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)
    model.initialize(opt, labeled_dataset, unlabeled_dataset)
    print("model [%s] was created." % (model.name()))
    return model