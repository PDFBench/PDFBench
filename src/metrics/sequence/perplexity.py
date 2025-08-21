accelerator: Accelerator = Accelerator(
            cpu=True,
        )

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.config.repeat.batch_size,  # FIXME:
            shuffle=False,
        )
        dataloader = accelerator.prepare(dataloader)

        results = []
        for batch in dataloader:
            res = [
                {
                    "instruction": item["instruction"],
                    "reference": item["reference"],
                }
                for item in batch
            ]

            for idx in range(len(batch)):
                for b in range(1, design_batch_size + 1):
                    response = batch[idx][f"response#{b}"]
                    res[idx].update(
                        {
                            f"response#{b}": response,
                            f"repeat_2#{b}": compute_repeatN(response, 2),
                            f"repeat_5#{b}": compute_repeatN(response, 5),
                            f"repeat#{b}": compute_repeat(response),
                        }
                    )
            results.extend(res)

        all_results = accelerator.gather(results)

        if accelerator.is_main_process:
            return EvaluationOutput(
                results=all_results,
                metrics=self.config.repeat.metrics,
                design_batch_size=self.config.basic.design_batch_size,
            )