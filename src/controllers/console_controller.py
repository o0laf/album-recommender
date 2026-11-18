"""Controlador de consola: recibe el comando que escribió la persona y se lo pasa al servicio que corresponde.
No tiene lógica de negocio. Los imports van adentro de cada método: así "preprocess" no carga TensorFlow."""


class ConsoleController:

    def run(self, args):
        handlers = {
            "preprocess": self.preprocess,
            "baseline": self.baseline,
            "train": self.train,
            "evaluate": self.evaluate,
            "experiments": self.experiments,
            "names": self.names,
            "recommend": self.recommend,
            "server": self.server,
            "coverage": self.coverage,
            "import-legacy": self.import_legacy,
        }
        handlers[args.command](args)

    def preprocess(self, args):
        from services.ratings_dataset_service import RatingsDatasetService
        RatingsDatasetService().build_data()

    def baseline(self, args):
        from services.recommendation_evaluation_service import RecommendationEvaluationService
        RecommendationEvaluationService().run_baseline()

    def train(self, args):
        from services.album_model_training_service import AlbumModelTrainingService
        AlbumModelTrainingService().run()

    def evaluate(self, args):
        from services.recommendation_evaluation_service import RecommendationEvaluationService
        RecommendationEvaluationService().run_evaluation()

    def experiments(self, args):
        from config import MAX_EPOCHS
        from services.training_experiment_service import TrainingExperimentService
        TrainingExperimentService().run(solo=args.solo, max_epochs=args.max_epochs or MAX_EPOCHS)

    def names(self, args):
        from services.album_catalog_service import AlbumCatalogService
        AlbumCatalogService().build_catalog()

    def recommend(self, args):
        from services.album_recommendation_service import AlbumRecommendationService
        AlbumRecommendationService().run_cli(args.albums, args.k)

    def server(self, args):
        from controllers.recommendation_web_controller import RecommendationWebController
        RecommendationWebController().run()

    def coverage(self, args):
        from services.vynl_coverage_service import VynlCoverageService
        VynlCoverageService().run(args.sqlite)

    def import_legacy(self, args):
        from services.legacy_artifacts_import_service import LegacyArtifactsImportService
        LegacyArtifactsImportService().run()
