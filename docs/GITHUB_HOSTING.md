# GitHub and hosted demo

The GitHub repository contains the reproducible Python project and the static Stockdesk demo. Notebook source is included. The local executed notebooks and their original outputs stay available in the workspace. Study results in the hosted build use the verified CA_1 aggregate simulation. The interactive order desk in that build uses synthetic demand and illustrative order targets. The exact historical item replay can be regenerated locally with the M5 inputs.

## Build and check the hosted version

```sh
node demo/build-site.mjs
python3 -m http.server 8771 --bind 127.0.0.1 --directory demo/site-build
```

The build has no package installation step. Deploy the static assets only. The Python order recommendation engine remains a separate batch process documented in [the batch guide](BATCH_DEPLOYMENT.md).

## Render

Connect the single GitHub repository in Render, then create a **Static Site**, or import the supplied `render.yaml` as a Blueprint. If using the manual form, select the `main` branch, leave **Root Directory** at the repository root, enter `node demo/build-site.mjs` as the build command, and set **Publish Directory** to `demo/site-build`. Add `SKIP_INSTALL_DEPS=true` if setting the fields manually. The resulting site uses an `onrender.com` URL. [Render Static Sites](https://render.com/docs/static-sites)

## Vercel

Import the GitHub repository with its root left at the repository root. The supplied `vercel.json` defines the static build and output. The equivalent settings are **Other** for the framework preset, no install command, build command `node demo/build-site.mjs`, and output `demo/site-build`. [Vercel build configuration](https://vercel.com/docs/builds/configure-a-build)

## Refresh the study after a research rerun

After recreating the verified forecast and inventory artifacts, run `python demo/export_data.py` to rebuild the local original replay, then `node demo/export_public.mjs` to update the hosted dataset. Test the hosted build again. The local historical replay export is ignored by Git.

To keep fresh notebook outputs local in another checkout, configure the supplied Git clean filter before adding notebook changes.

```sh
git config filter.stockdesk-notebook.clean 'python3 tools/strip_notebook_outputs.py'
git config filter.stockdesk-notebook.smudge cat
git config filter.stockdesk-notebook.required true
```

