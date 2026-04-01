writeup_agent_prompt_template = """Your goal is to write up the following idea:

```markdown
{research_overview_text}
```
Note that idea_text represents a preliminary hypothesis and may not necessarily align with the experiments that were eventually performed.

First, make sure to refer to the experiment data contained in table/ and figure/ folders.


We have VLM-based table descriptions:
```
{table_descriptions}
```

We also have VLM-based figure descriptions:
```
{plot_descriptions}
```

To better understand the methodology and experiments, please also refer to:
- code/ as the proposed method's code implementation

Please read the current template.tex file and update it to produce a complete, coherent, and scientifically accurate paper.
This must be an acceptable complete LaTeX writeup, suitable for a {num_page}-page {column_type} paper.
Make sure to use the citations from the references.bib file and report results accurately based on the experimental data provided.
IMPORTANT: references.bib can be very large. Do NOT read the entire file at once. Instead, use Grep to search for relevant citation keys or authors, then Read only the specific portions you need (using offset and limit parameters).
Start by reading template.tex to understand the current state, then edit it to incorporate all the information above into a complete paper.


Please note: For the bibliography, do not use the \\begin{{filecontents}}{{references.bib}} environment. Instead, all citations should refer to an external file named references.bib."""


reflection_prompt_template = """Now let's reflect and identify any issues (including but not limited to).
Your task is to read the current template.tex file and improve it based on the feedback provided below.
1) Are there any LaTeX syntax errors or style violations we can fix? Refer to the chktex output below.

chktex results:
```
{check_output}
```

2) Are there any LaTeX compilation errors? Refer to the tectonic compile output below.

tectonic compile output:
```
{compile_output}
```

If there are errors reported above, please fix them directly.
Read template.tex and edit it to address these issues.
Focus especially on fixing compilation errors so that the paper compiles successfully.

If no errors are reported, no changes are necessary."""


page_limit_prompt_template = """The main text (before 'References') is currently {main_pages} pages. The target is {page_limit} pages.

The paper is {status}. Please {action} to reach the target.
Do NOT move content to or create an Appendix. Keep everything in the main text.
Do not add or remove more than 1000 characters in this revision. Do not use \\begin{{filecontents}}{{references.bib}}."""
